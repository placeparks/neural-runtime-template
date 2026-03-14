const fs = require("fs");
const os = require("os");
const path = require("path");
const { Readable } = require("stream");
const { pipeline } = require("stream/promises");
const { spawn } = require("child_process");

const { Client, GatewayIntentBits, Partials } = require("discord.js");
const {
  AudioPlayerStatus,
  EndBehaviorType,
  StreamType,
  entersState,
  joinVoiceChannel,
  VoiceConnectionStatus,
  createAudioPlayer,
  createAudioResource,
} = require("@discordjs/voice");
const prism = require("prism-media");

const DISCORD_TOKEN = (process.env.NEURALCLAW_DISCORD_TOKEN || "").trim();
const OPENAI_API_KEY = (process.env.NEURALCLAW_VOICE_OPENAI_KEY || process.env.OPENAI_API_KEY || "").trim();
const PORT = (process.env.PORT || process.env.NEURALCLAW_MESH_PORT || "8100").trim();
const GATEWAY_URL = (process.env.NEURALCLAW_DISCORD_GATEWAY_URL || `http://127.0.0.1:${PORT}`).trim().replace(/\/$/, "");
const SHARED_SECRET = (process.env.NEURALCLAW_MESH_SHARED_SECRET || "").trim();
const TTS_MODEL = (process.env.NEURALCLAW_DISCORD_TTS_MODEL || "gpt-4o-mini-tts").trim();
const TTS_VOICE = (process.env.NEURALCLAW_DISCORD_TTS_VOICE || "alloy").trim();
const TTS_INSTRUCTIONS = (
  process.env.NEURALCLAW_DISCORD_TTS_INSTRUCTIONS ||
  "Speak naturally, warmly, and conversationally. Sound human and responsive."
).trim();
const TRANSCRIBE_MODEL = (process.env.NEURALCLAW_DISCORD_TRANSCRIBE_MODEL || "gpt-4o-mini-transcribe").trim();
const VOICE_ENABLED = /^(1|true|yes|on)$/i.test(process.env.NEURALCLAW_DISCORD_VOICE_ENABLED || "false");
const VOICE_REPLY_WITH_TEXT = /^(1|true|yes|on)$/i.test(
  process.env.NEURALCLAW_DISCORD_VOICE_REPLY_WITH_TEXT || "true"
);
const SILENCE_MS = Math.max(300, parseInt(process.env.NEURALCLAW_DISCORD_VOICE_SILENCE_MS || "900", 10) || 900);
const MIN_SEGMENT_MS = Math.max(200, parseInt(process.env.NEURALCLAW_DISCORD_VOICE_MIN_MS || "500", 10) || 500);
const MAX_SEGMENT_SECONDS = Math.max(3, parseInt(process.env.NEURALCLAW_DISCORD_VOICE_MAX_SECONDS || "15", 10) || 15);

if (!DISCORD_TOKEN) {
  console.error("[DiscordVoice] NEURALCLAW_DISCORD_TOKEN is not configured");
  process.exit(1);
}

const client = new Client({
  intents: [
    GatewayIntentBits.Guilds,
    GatewayIntentBits.GuildMessages,
    GatewayIntentBits.DirectMessages,
    GatewayIntentBits.MessageContent,
    GatewayIntentBits.GuildVoiceStates,
  ],
  partials: [Partials.Channel],
});

const sessions = new Map();

function isJoinCommand(text) {
  return [
    "join voice",
    "join voice channel",
    "join my voice channel",
    "join vc",
    "come to voice",
    "come to my voice channel",
    "join me in voice",
  ].includes(text);
}

function isLeaveCommand(text) {
  return [
    "leave voice",
    "leave voice channel",
    "leave vc",
    "disconnect",
    "disconnect voice",
  ].includes(text);
}

function stripMention(content) {
  if (!client.user) {
    return content.trim();
  }
  return content
    .replaceAll(`<@${client.user.id}>`, "")
    .replaceAll(`<@!${client.user.id}>`, "")
    .trim();
}

async function callGateway(content, meta) {
  const headers = { "Content-Type": "application/json" };
  if (SHARED_SECRET) {
    headers["x-mesh-secret"] = SHARED_SECRET;
  }

  const payload = {
    content,
    from: meta.authorId,
    author_name: meta.authorName,
    channel_id: meta.channelId,
    channel_type_name: meta.channelTypeName,
    message_metadata: meta.messageMetadata,
  };

  const resp = await fetch(`${GATEWAY_URL}/a2a/message`, {
    method: "POST",
    headers,
    body: JSON.stringify(payload),
  });
  const raw = await resp.text();
  if (!resp.ok) {
    throw new Error(`gateway request failed (${resp.status}): ${raw.slice(0, 300)}`);
  }
  const data = JSON.parse(raw);
  return String(data.content || "").trim();
}

async function transcribeWavBytes(wavBuffer) {
  if (!OPENAI_API_KEY) {
    throw new Error("No OpenAI key configured for Discord voice transcription.");
  }

  const form = new FormData();
  form.append("model", TRANSCRIBE_MODEL);
  form.append("file", new Blob([wavBuffer], { type: "audio/wav" }), "discord-voice.wav");

  const resp = await fetch("https://api.openai.com/v1/audio/transcriptions", {
    method: "POST",
    headers: { Authorization: `Bearer ${OPENAI_API_KEY}` },
    body: form,
  });
  const raw = await resp.text();
  if (!resp.ok) {
    throw new Error(`OpenAI transcription failed (${resp.status}): ${raw.slice(0, 300)}`);
  }
  const data = JSON.parse(raw);
  const text = String(data.text || "").trim();
  if (!text) {
    throw new Error("OpenAI transcription returned empty text.");
  }
  return text;
}

async function synthesizeSpeech(text) {
  if (!OPENAI_API_KEY) {
    throw new Error("No OpenAI key configured for Discord TTS.");
  }
  const clean = text.trim().slice(0, 4000);
  const resp = await fetch("https://api.openai.com/v1/audio/speech", {
    method: "POST",
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model: TTS_MODEL,
      voice: TTS_VOICE,
      input: clean,
      instructions: TTS_INSTRUCTIONS,
      format: "mp3",
    }),
  });
  const arr = await resp.arrayBuffer();
  if (!resp.ok) {
    const detail = Buffer.from(arr).toString("utf8").slice(0, 300);
    throw new Error(`OpenAI speech failed (${resp.status}): ${detail}`);
  }
  const buf = Buffer.from(arr);
  if (!buf.length) {
    throw new Error("OpenAI speech returned empty audio.");
  }
  return buf;
}

function pcmToWav(pcmBuffer) {
  const sampleRate = 48000;
  const channels = 2;
  const bitDepth = 16;
  const blockAlign = (channels * bitDepth) / 8;
  const byteRate = sampleRate * blockAlign;
  const out = Buffer.alloc(44 + pcmBuffer.length);
  out.write("RIFF", 0);
  out.writeUInt32LE(36 + pcmBuffer.length, 4);
  out.write("WAVE", 8);
  out.write("fmt ", 12);
  out.writeUInt32LE(16, 16);
  out.writeUInt16LE(1, 20);
  out.writeUInt16LE(channels, 22);
  out.writeUInt32LE(sampleRate, 24);
  out.writeUInt32LE(byteRate, 28);
  out.writeUInt16LE(blockAlign, 32);
  out.writeUInt16LE(bitDepth, 34);
  out.write("data", 36);
  out.writeUInt32LE(pcmBuffer.length, 40);
  pcmBuffer.copy(out, 44);
  return out;
}

function pcmBytesForMs(ms) {
  return Math.floor(48000 * 2 * 2 * (ms / 1000));
}

async function mp3ToPcmResource(mp3Buffer) {
  const ffmpeg = new prism.FFmpeg({
    args: [
      "-analyzeduration",
      "0",
      "-loglevel",
      "0",
      "-i",
      "pipe:0",
      "-f",
      "s16le",
      "-ar",
      "48000",
      "-ac",
      "2",
      "pipe:1",
    ],
  });
  Readable.from(mp3Buffer).pipe(ffmpeg);
  return createAudioResource(ffmpeg, { inputType: StreamType.Raw });
}

async function playSpeech(guildId, text) {
  const session = sessions.get(guildId);
  if (!session) {
    return;
  }
  const mp3 = await synthesizeSpeech(text);
  const resource = await mp3ToPcmResource(mp3);
  const player = session.player;

  const done = new Promise((resolve, reject) => {
    const cleanup = () => {
      player.off(AudioPlayerStatus.Idle, onIdle);
      player.off("error", onError);
    };
    const onIdle = () => {
      cleanup();
      resolve();
    };
    const onError = (err) => {
      cleanup();
      reject(err);
    };
    player.once(AudioPlayerStatus.Idle, onIdle);
    player.once("error", onError);
  });

  player.play(resource);
  await done;
}

async function enqueueSpeech(guildId, text) {
  const session = sessions.get(guildId);
  if (!session) {
    return;
  }
  session.queue = session.queue.then(async () => {
    await playSpeech(guildId, text);
    if (VOICE_REPLY_WITH_TEXT && session.controlChannelId) {
      const channel = await client.channels.fetch(session.controlChannelId).catch(() => null);
      if (channel && channel.isTextBased()) {
        await channel.send(text).catch((err) => {
          console.warn("[DiscordVoice] failed to send text echo:", err.message);
        });
      }
    }
  }).catch((err) => {
    console.warn("[DiscordVoice] voice playback failed:", err.message);
  });
  await session.queue;
}

async function processVoiceSegment(session, user, pcmBuffer) {
  if (!pcmBuffer.length || pcmBuffer.length < pcmBytesForMs(MIN_SEGMENT_MS)) {
    return;
  }
  const wav = pcmToWav(pcmBuffer);
  const transcript = await transcribeWavBytes(wav);
  console.log(
    `[DiscordVoice] inbound voice segment guild=${session.guildId} user=${user.id} transcript_len=${transcript.length}`
  );
  const response = await callGateway(transcript, {
    authorId: String(user.id),
    authorName: user.displayName || user.username || `user-${user.id}`,
    channelId: `discord-voice:${session.guildId}`,
    channelTypeName: "DISCORD",
    messageMetadata: {
      platform: "discord",
      source: "discord",
      discord_voice_reply: true,
      discord_guild_id: String(session.guildId),
      discord_guild_name: session.guildName,
      discord_voice_channel_id: String(session.voiceChannelId),
      discord_control_channel_id: String(session.controlChannelId || ""),
    },
  });
  if (!response) {
    return;
  }
  await enqueueSpeech(session.guildId, response);
}

function subscribeToUser(session, userId) {
  if (session.subscriptions.has(userId)) {
    return;
  }

  const opusStream = session.connection.receiver.subscribe(userId, {
    end: {
      behavior: EndBehaviorType.AfterSilence,
      duration: SILENCE_MS,
    },
  });
  const decoder = new prism.opus.Decoder({ frameSize: 960, channels: 2, rate: 48000 });
  const chunks = [];
  let total = 0;
  const maxBytes = 48000 * 2 * 2 * MAX_SEGMENT_SECONDS;

  session.subscriptions.set(userId, opusStream);

  const cleanup = () => {
    session.subscriptions.delete(userId);
    opusStream.destroy();
    decoder.destroy();
  };

  opusStream.on("error", (err) => {
    console.warn(`[DiscordVoice] opus receive error guild=${session.guildId} user=${userId}: ${err.message}`);
    cleanup();
  });
  decoder.on("error", (err) => {
    console.warn(`[DiscordVoice] decode error guild=${session.guildId} user=${userId}: ${err.message}`);
    cleanup();
  });

  decoder.on("data", (chunk) => {
    chunks.push(chunk);
    total += chunk.length;
    if (total >= maxBytes) {
      opusStream.emit("end");
    }
  });

  decoder.once("close", async () => {
    cleanup();
    const pcm = Buffer.concat(chunks, total);
    const user = await client.users.fetch(userId).catch(() => null);
    if (!user) {
      return;
    }
    try {
      await processVoiceSegment(session, user, pcm);
    } catch (err) {
      console.warn(
        `[DiscordVoice] voice segment processing failed guild=${session.guildId} user=${userId}: ${err.message}`
      );
      if (session.controlChannelId) {
        const channel = await client.channels.fetch(session.controlChannelId).catch(() => null);
        if (channel && channel.isTextBased()) {
          await channel.send(`I couldn't process that voice segment: ${err.message}`).catch(() => {});
        }
      }
    }
  });

  opusStream.pipe(decoder);
}

async function joinVoice(message) {
  const voiceChannel = message.member?.voice?.channel;
  if (!voiceChannel) {
    await message.channel.send("Join a voice channel first, then ask me to join.");
    return;
  }
  if (!VOICE_ENABLED) {
    await message.channel.send("Discord voice is not enabled for this agent.");
    return;
  }

  const existing = sessions.get(message.guild.id);
  if (existing) {
    if (existing.voiceChannelId !== voiceChannel.id) {
      existing.connection.destroy();
      sessions.delete(message.guild.id);
    } else {
      existing.controlChannelId = message.channel.id;
      await message.channel.send(`Already in voice channel \`${voiceChannel.name}\`.`);
      return;
    }
  }

  const connection = await createReliableVoiceConnection(voiceChannel);
  const player = createAudioPlayer();
  connection.subscribe(player);

  const session = {
    guildId: message.guild.id,
    guildName: message.guild.name,
    voiceChannelId: voiceChannel.id,
    controlChannelId: message.channel.id,
    connection,
    player,
    subscriptions: new Map(),
    queue: Promise.resolve(),
  };
  sessions.set(message.guild.id, session);

  connection.receiver.speaking.on("start", (userId) => {
    if (userId === client.user.id) {
      return;
    }
    subscribeToUser(session, userId);
  });

  connection.on("stateChange", async (_, newState) => {
    if (newState.status === VoiceConnectionStatus.Destroyed || newState.status === VoiceConnectionStatus.Disconnected) {
      sessions.delete(message.guild.id);
    }
  });

  console.log(`[DiscordVoice] voice session started guild=${message.guild.id} channel=${voiceChannel.id}`);
  await message.channel.send(`Joined voice channel \`${voiceChannel.name}\`. Speak normally and I'll answer in voice.`);
}

async function createReliableVoiceConnection(voiceChannel) {
  let lastError = null;

  for (let attempt = 1; attempt <= 3; attempt += 1) {
    const connection = joinVoiceChannel({
      channelId: voiceChannel.id,
      guildId: voiceChannel.guild.id,
      adapterCreator: voiceChannel.guild.voiceAdapterCreator,
      selfDeaf: false,
      selfMute: false,
    });

    connection.on("stateChange", (oldState, newState) => {
      console.log(
        `[DiscordVoice] connection state guild=${voiceChannel.guild.id} ${oldState.status} -> ${newState.status}`
      );
    });

    try {
      await entersState(connection, VoiceConnectionStatus.Ready, 45_000);
      return connection;
    } catch (err) {
      lastError = err;
      const status = connection.state?.status || "unknown";
      const detail = err?.message || String(err);
      console.warn(
        `[DiscordVoice] voice connect attempt ${attempt} failed guild=${voiceChannel.guild.id} status=${status}: ${detail}`
      );

      try {
        connection.destroy();
      } catch (_) {}

      if (attempt < 3) {
        await new Promise((resolve) => setTimeout(resolve, 1500 * attempt));
      }
    }
  }

  const detail = lastError?.message || String(lastError || "unknown error");
  throw new Error(`Voice connection failed after retries: ${detail}`);
}

async function leaveVoice(message) {
  const session = sessions.get(message.guild?.id);
  if (!session) {
    await message.channel.send("I'm not in a voice channel right now.");
    return;
  }
  session.connection.destroy();
  sessions.delete(message.guild.id);
  await message.channel.send("Left the voice channel.");
}

client.on("ready", () => {
  console.log(`[DiscordVoice] Logged in as ${client.user?.tag || "unknown"}`);
});

client.on("messageCreate", async (message) => {
  if (message.author.bot) {
    return;
  }

  const isDm = !message.guild;
  const mentioned = client.user ? message.mentions.has(client.user) : false;
  if (!isDm && !mentioned) {
    return;
  }

  const content = stripMention(message.content || "");
  const lowered = content.toLowerCase();

  try {
    if (!isDm && VOICE_ENABLED && isJoinCommand(lowered)) {
      await joinVoice(message);
      return;
    }
    if (!isDm && VOICE_ENABLED && isLeaveCommand(lowered)) {
      await leaveVoice(message);
      return;
    }

    const response = await callGateway(content, {
      authorId: String(message.author.id),
      authorName: message.member?.displayName || message.author.username,
      channelId: String(message.channel.id),
      channelTypeName: "DISCORD",
      messageMetadata: {
        platform: "discord",
        source: "discord",
        is_dm: isDm,
        guild: message.guild?.name || null,
      },
    });
    if (response) {
      await message.channel.send(response);
    }
  } catch (err) {
    console.error("[DiscordVoice] message handling failed:", err);
    await message.channel.send(`I hit an error: ${err.message}`).catch(() => {});
  }
});

client.on("voiceStateUpdate", async (before, after) => {
  if (!client.user || !before.guild) {
    return;
  }
  if (before.member?.id !== client.user.id) {
    return;
  }
  if (before.channelId && !after.channelId) {
    sessions.delete(before.guild.id);
  }
});

async function main() {
  await client.login(DISCORD_TOKEN);
}

main().catch((err) => {
  console.error("[DiscordVoice] fatal error:", err);
  process.exit(1);
});
