const fs = require("fs");
const os = require("os");
const path = require("path");
const { PassThrough, Readable } = require("stream");
const WebSocket = require("ws");

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
  demuxProbe,
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
  "Speak naturally, warmly, and conversationally. Sound human, playful, emotionally expressive, and responsive. Be helpful without sounding submissive or robotic."
).trim();
const TRANSCRIBE_MODEL = (process.env.NEURALCLAW_DISCORD_TRANSCRIBE_MODEL || "gpt-4o-transcribe").trim();
const TRANSCRIBE_LANGUAGE = (process.env.NEURALCLAW_DISCORD_TRANSCRIBE_LANGUAGE || "en").trim();
const TRANSCRIBE_PROMPT = (
  process.env.NEURALCLAW_DISCORD_TRANSCRIBE_PROMPT ||
  "The speaker is primarily speaking conversational English. Prefer a faithful transcription over paraphrasing."
).trim();
const VOICE_ENABLED = /^(1|true|yes|on)$/i.test(process.env.NEURALCLAW_DISCORD_VOICE_ENABLED || "false");
const VOICE_REPLY_WITH_TEXT = /^(1|true|yes|on)$/i.test(
  process.env.NEURALCLAW_DISCORD_VOICE_REPLY_WITH_TEXT || "true"
);
const SILENCE_MS = Math.max(500, parseInt(process.env.NEURALCLAW_DISCORD_VOICE_SILENCE_MS || "1400", 10) || 1400);
const MIN_SEGMENT_MS = Math.max(400, parseInt(process.env.NEURALCLAW_DISCORD_VOICE_MIN_MS || "900", 10) || 900);
const MAX_SEGMENT_SECONDS = Math.max(5, parseInt(process.env.NEURALCLAW_DISCORD_VOICE_MAX_SECONDS || "18", 10) || 18);
const REALTIME_ENABLED = /^(1|true|yes|on)$/i.test(process.env.NEURALCLAW_DISCORD_REALTIME_ENABLED || "true");
const REALTIME_MODEL = (process.env.NEURALCLAW_DISCORD_REALTIME_MODEL || process.env.NEURALCLAW_VOICE_REALTIME_MODEL || "gpt-realtime").trim();
const REALTIME_VOICE = (process.env.NEURALCLAW_DISCORD_REALTIME_VOICE || process.env.NEURALCLAW_VOICE_REALTIME_VOICE || "coral").trim();

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

const STOP_COMMAND_RE = /\b(stop|pause|wait|hold on|hold up|quiet|be quiet|stop talking|shut up)\b/i;

function buildRealtimeInstructions() {
  const persona = (process.env.NEURALCLAW_VOICE_PERSONA || process.env.NEURALCLAW_PERSONA || "You are NeuralClaw, a helpful and intelligent AI assistant.").trim();
  return [
    persona,
    "You are speaking live in a Discord voice call.",
    "Sound natural, warm, concise, emotionally expressive, and human.",
    "Keep replies short by default, usually 1 to 3 sentences.",
    "Have a distinct, likable personality. Be playful and lightly teasing at times when it feels natural, but never mean or dismissive.",
    "You may laugh softly or react with warmth when it genuinely fits the moment, but do not overdo it or force catchphrases.",
    "Be helpful and engaging, but do not be blindly agreeable. If the user is wrong, say so kindly, clearly, and confidently.",
    "Act like a sharp, socially aware human conversation partner, not a scripted assistant.",
    "Always respond in English unless the user very clearly asks you to switch languages.",
    "If the audio is unclear, ask for a repeat in simple English.",
    "Do not infer another language from noisy or partial audio.",
    "If the user interrupts, stop and listen immediately.",
    "Do not use markdown, lists, or emoji while speaking.",
  ].join("\n\n");
}

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
  if (TRANSCRIBE_LANGUAGE) {
    form.append("language", TRANSCRIBE_LANGUAGE);
  }
  if (TRANSCRIBE_PROMPT) {
    form.append("prompt", TRANSCRIBE_PROMPT);
  }
  form.append("file", new Blob([wavBuffer], { type: "audio/wav" }), "discord-voice.wav");

  console.log(`[DiscordVoice] transcription start bytes=${wavBuffer.length} model=${TRANSCRIBE_MODEL} language=${TRANSCRIBE_LANGUAGE || "auto"}`);
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
  console.log(`[DiscordVoice] transcription success chars=${text.length}`);
  return text;
}

async function synthesizeSpeech(text) {
  if (!OPENAI_API_KEY) {
    throw new Error("No OpenAI key configured for Discord TTS.");
  }
  const clean = text.trim().slice(0, 4000);
  console.log(`[DiscordVoice] tts start chars=${clean.length} model=${TTS_MODEL} voice=${TTS_VOICE}`);
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
  console.log(`[DiscordVoice] tts success bytes=${buf.length}`);
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

function discordPcmToRealtimePcm(pcmBuffer) {
  const frameCount = Math.floor(pcmBuffer.length / 4);
  const out = Buffer.allocUnsafe(frameCount * 2);
  let outOffset = 0;
  for (let i = 0; i < frameCount; i += 2) {
    const frameOffset = i * 4;
    const left = pcmBuffer.readInt16LE(frameOffset);
    const right = pcmBuffer.readInt16LE(frameOffset + 2);
    const mono = Math.max(-32768, Math.min(32767, Math.round((left + right) / 2)));
    out.writeInt16LE(mono, outOffset);
    outOffset += 2;
  }
  return out.subarray(0, outOffset);
}

function realtimePcmToDiscordPcm(pcmBuffer) {
  const sampleCount = Math.floor(pcmBuffer.length / 2);
  const out = Buffer.allocUnsafe(sampleCount * 8);
  let outOffset = 0;
  for (let i = 0; i < sampleCount; i += 1) {
    const sample = pcmBuffer.readInt16LE(i * 2);
    for (let repeat = 0; repeat < 2; repeat += 1) {
      out.writeInt16LE(sample, outOffset);
      out.writeInt16LE(sample, outOffset + 2);
      outOffset += 4;
    }
  }
  return out;
}

function ensureRealtimePlaybackStream(session) {
  if (session.realtimePlaybackStream) {
    return session.realtimePlaybackStream;
  }
  const stream = new PassThrough();
  const resource = createAudioResource(stream, { inputType: StreamType.Raw });
  session.realtimePlaybackStream = stream;
  console.log(`[DiscordVoice] realtime playback request guild=${session.guildId}`);
  session.player.play(resource);
  return stream;
}

function closeRealtimePlaybackStream(session, reason = "done") {
  if (!session.realtimePlaybackStream) {
    return;
  }
  console.log(`[DiscordVoice] realtime playback close guild=${session.guildId} reason=${reason}`);
  try {
    session.realtimePlaybackStream.end();
  } catch (_) {}
  session.realtimePlaybackStream = null;
}

function sendRealtimeEvent(session, payload) {
  const ws = session.realtimeSocket;
  if (!ws || ws.readyState !== WebSocket.OPEN) {
    return;
  }
  ws.send(JSON.stringify(payload));
}

function interruptRealtimeResponse(session, reason) {
  if (!session.realtimeResponseActive && !session.realtimePlaybackStream) {
    return;
  }
  interruptPlayback(session, reason);
  closeRealtimePlaybackStream(session, reason);
  if (session.realtimeResponseActive) {
    sendRealtimeEvent(session, { type: "response.cancel" });
    session.realtimeResponseActive = false;
  }
}

function handleRealtimeEvent(session, rawData) {
  let event;
  try {
    event = JSON.parse(String(rawData));
  } catch (err) {
    console.warn(`[DiscordVoice] realtime parse error guild=${session.guildId}: ${err.message}`);
    return;
  }

  const etype = event.type || "";
  if (etype === "session.created" || etype === "session.updated") {
    return;
  }
  if (etype === "conversation.item.input_audio_transcription.completed") {
    const transcript = String(event.transcript || "").trim();
    if (transcript) {
      console.log(`[DiscordVoice] realtime transcript guild=${session.guildId} chars=${transcript.length}`);
    }
    return;
  }
  if (etype === "response.audio_transcript.done") {
    const transcript = String(event.transcript || "").trim();
    if (transcript && VOICE_REPLY_WITH_TEXT && session.controlChannelId) {
      client.channels.fetch(session.controlChannelId).then((channel) => {
        if (channel && channel.isTextBased()) {
          return channel.send(transcript);
        }
      }).catch((err) => {
        console.warn(`[DiscordVoice] realtime text echo failed guild=${session.guildId}: ${err.message}`);
      });
    }
    return;
  }
  if (etype === "response.created") {
    session.realtimeResponseActive = true;
    return;
  }
  if (etype === "response.audio.delta") {
    session.realtimeResponseActive = true;
    const delta = String(event.delta || "");
    if (!delta) {
      return;
    }
    const pcm24 = Buffer.from(delta, "base64");
    const discordPcm = realtimePcmToDiscordPcm(pcm24);
    ensureRealtimePlaybackStream(session).write(discordPcm);
    return;
  }
  if (etype === "response.done" || etype === "response.audio.done") {
    session.realtimeResponseActive = false;
    closeRealtimePlaybackStream(session, etype);
    return;
  }
  if (etype === "input_audio_buffer.speech_started") {
    if (session.realtimeResponseActive || session.realtimePlaybackStream) {
      interruptPlayback(session, "realtime_speech_started");
      closeRealtimePlaybackStream(session, "speech_started");
    }
    return;
  }
  if (etype === "error") {
    if (event.error && event.error.code === "response_cancel_not_active") {
      session.realtimeResponseActive = false;
      return;
    }
    console.warn(`[DiscordVoice] realtime error guild=${session.guildId}: ${JSON.stringify(event.error || event)}`);
  }
}

async function openRealtimeSession(session) {
  if (!OPENAI_API_KEY) {
    throw new Error("No OpenAI key configured for Discord Realtime voice.");
  }
  const url = `wss://api.openai.com/v1/realtime?model=${encodeURIComponent(REALTIME_MODEL)}`;
  console.log(`[DiscordVoice] realtime connect guild=${session.guildId} model=${REALTIME_MODEL} voice=${REALTIME_VOICE}`);
  const ws = new WebSocket(url, {
    headers: {
      Authorization: `Bearer ${OPENAI_API_KEY}`,
      "OpenAI-Beta": "realtime=v1",
    },
  });

  session.realtimeSocket = ws;

  await new Promise((resolve, reject) => {
    const onOpen = () => {
      ws.off("error", onError);
      resolve();
    };
    const onError = (err) => {
      ws.off("open", onOpen);
      reject(err);
    };
    ws.once("open", onOpen);
    ws.once("error", onError);
  });

  ws.on("message", (data) => handleRealtimeEvent(session, data));
  ws.on("close", () => {
    console.log(`[DiscordVoice] realtime close guild=${session.guildId}`);
    closeRealtimePlaybackStream(session, "socket_closed");
    if (session.realtimeSocket === ws) {
      session.realtimeSocket = null;
    }
  });
  ws.on("error", (err) => {
    console.warn(`[DiscordVoice] realtime socket error guild=${session.guildId}: ${err.message}`);
  });

  sendRealtimeEvent(session, {
    type: "session.update",
    session: {
      turn_detection: {
        type: "server_vad",
        create_response: false,
        interrupt_response: true,
      },
      input_audio_format: "pcm16",
      output_audio_format: "pcm16",
      voice: REALTIME_VOICE,
      instructions: buildRealtimeInstructions(),
      modalities: ["text", "audio"],
      temperature: 0.8,
      input_audio_transcription: {
        model: TRANSCRIBE_MODEL,
      },
    },
  });
  console.log(`[DiscordVoice] realtime ready guild=${session.guildId}`);
}

async function mp3ToAudioResource(mp3Buffer) {
  console.log(`[DiscordVoice] resource probe start bytes=${mp3Buffer.length}`);
  const stream = new PassThrough();
  stream.end(mp3Buffer);
  const probed = await demuxProbe(stream);
  console.log(`[DiscordVoice] resource probe success type=${probed.type}`);
  return createAudioResource(probed.stream, { inputType: probed.type });
}

async function playSpeech(guildId, text) {
  const session = sessions.get(guildId);
  if (!session) {
    return { interrupted: false };
  }
  const generation = session.playbackGeneration;
  const mp3 = await synthesizeSpeech(text);
  if (generation !== session.playbackGeneration) {
    console.log(`[DiscordVoice] playback skipped guild=${guildId} reason=interrupted_before_probe`);
    return { interrupted: true };
  }
  const resource = await mp3ToAudioResource(mp3);
  const player = session.player;

  const done = new Promise((resolve, reject) => {
    const cleanup = () => {
      player.off(AudioPlayerStatus.Idle, onIdle);
      player.off("error", onError);
    };
    const onIdle = () => {
      cleanup();
      resolve({ interrupted: generation !== session.playbackGeneration });
    };
    const onError = (err) => {
      cleanup();
      reject(err);
    };
    player.once(AudioPlayerStatus.Idle, onIdle);
    player.once("error", onError);
  });

  console.log(`[DiscordVoice] playback request guild=${guildId}`);
  player.play(resource);
  const result = await done;
  console.log(`[DiscordVoice] playback end guild=${guildId} interrupted=${result.interrupted ? "true" : "false"}`);
  return result;
}

async function enqueueSpeech(guildId, text) {
  const session = sessions.get(guildId);
  if (!session) {
    return;
  }
  const generation = session.playbackGeneration;
  session.queue = session.queue.then(async () => {
    if (!sessions.get(guildId) || generation !== session.playbackGeneration) {
      console.log(`[DiscordVoice] queued reply dropped guild=${guildId} reason=interrupted_before_play`);
      return;
    }
    const playback = await playSpeech(guildId, text);
    if (!playback.interrupted && VOICE_REPLY_WITH_TEXT && session.controlChannelId) {
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

function interruptPlayback(session, reason) {
  session.playbackGeneration += 1;
  console.log(`[DiscordVoice] interrupt playback guild=${session.guildId} reason=${reason}`);
  try {
    if (session.player && session.player.state && session.player.state.status !== AudioPlayerStatus.Idle) {
      session.player.stop(true);
    }
  } catch (err) {
    console.warn(`[DiscordVoice] interrupt playback error guild=${session.guildId}: ${err.message}`);
  }
}

async function processVoiceSegment(session, user, pcmBuffer) {
  if (session.realtimeEnabled) {
    return;
  }
  if (!pcmBuffer.length || pcmBuffer.length < pcmBytesForMs(MIN_SEGMENT_MS)) {
    return;
  }
  const wav = pcmToWav(pcmBuffer);
  const transcript = await transcribeWavBytes(wav);
  console.log(
    `[DiscordVoice] inbound voice segment guild=${session.guildId} user=${user.id} transcript_len=${transcript.length}`
  );
  if (STOP_COMMAND_RE.test(transcript)) {
    interruptPlayback(session, "voice_stop_command");
    return;
  }
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
    if (session.realtimeEnabled && session.realtimeSocket && session.realtimeSocket.readyState === WebSocket.OPEN) {
      const realtimePcm = discordPcmToRealtimePcm(chunk);
      if (realtimePcm.length > 0) {
        sendRealtimeEvent(session, {
          type: "input_audio_buffer.append",
          audio: realtimePcm.toString("base64"),
        });
      }
    }
    chunks.push(chunk);
    total += chunk.length;
    if (total >= maxBytes) {
      opusStream.emit("end");
    }
  });

  decoder.once("close", async () => {
    cleanup();
    const pcm = Buffer.concat(chunks, total);
    console.log(`[DiscordVoice] segment close guild=${session.guildId} user=${userId} pcm_bytes=${pcm.length}`);
    if (session.realtimeEnabled) {
      if (pcm.length >= pcmBytesForMs(MIN_SEGMENT_MS) && session.realtimeSocket && session.realtimeSocket.readyState === WebSocket.OPEN) {
        console.log(`[DiscordVoice] realtime commit guild=${session.guildId} user=${userId} bytes=${pcm.length}`);
        sendRealtimeEvent(session, { type: "input_audio_buffer.commit" });
        sendRealtimeEvent(session, {
          type: "response.create",
          response: {
            modalities: ["audio", "text"],
          },
        });
      }
      return;
    }
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
  player.on("stateChange", (oldState, newState) => {
    console.log(
      `[DiscordVoice] player state guild=${message.guild.id} ${oldState.status} -> ${newState.status}`
    );
  });
  player.on("error", (err) => {
    console.warn(`[DiscordVoice] player error guild=${message.guild.id}: ${err.message}`);
  });
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
    playbackGeneration: 0,
    realtimeEnabled: REALTIME_ENABLED,
    realtimeSocket: null,
    realtimePlaybackStream: null,
    realtimeResponseActive: false,
  };
  sessions.set(message.guild.id, session);

  if (session.realtimeEnabled) {
    try {
      await openRealtimeSession(session);
    } catch (err) {
      sessions.delete(message.guild.id);
      connection.destroy();
      throw err;
    }
  }

  connection.receiver.speaking.on("start", (userId) => {
    if (userId === client.user.id) {
      return;
    }
    if (session.realtimeEnabled) {
      if (session.realtimeResponseActive || session.realtimePlaybackStream) {
        interruptRealtimeResponse(session, `barge_in_${userId}`);
      }
    } else {
      interruptPlayback(session, `barge_in_${userId}`);
    }
    subscribeToUser(session, userId);
  });

  connection.on("stateChange", async (_, newState) => {
    if (newState.status === VoiceConnectionStatus.Destroyed || newState.status === VoiceConnectionStatus.Disconnected) {
      if (session.realtimeSocket) {
        try {
          session.realtimeSocket.close();
        } catch (_) {}
      }
      closeRealtimePlaybackStream(session, "connection_closed");
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
  if (session.realtimeSocket) {
    try {
      session.realtimeSocket.close();
    } catch (_) {}
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
