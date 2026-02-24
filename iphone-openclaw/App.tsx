import { Audio } from "expo-av";
import { useKeepAwake } from "expo-keep-awake";
import { LinearGradient } from "expo-linear-gradient";
import { StatusBar } from "expo-status-bar";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  ActivityIndicator,
  Pressable,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  View,
} from "react-native";

type Turn = {
  role: "user" | "assistant";
  text: string;
};

type TalkResponse = {
  session_id: string;
  transcript: string;
  reply_text: string;
  audio_url: string | null;
};

const AUTO_SPEECH_START_DB = -45;
const AUTO_SILENCE_MS = 900;
const AUTO_MAX_RECORD_MS = 7000;

function normalizeBaseUrl(input: string): string {
  return input.trim().replace(/\/+$/, "");
}

function absoluteUrl(baseUrl: string, maybePath: string | null): string | null {
  if (!maybePath) {
    return null;
  }
  if (maybePath.startsWith("http://") || maybePath.startsWith("https://")) {
    return maybePath;
  }
  return `${normalizeBaseUrl(baseUrl)}${maybePath}`;
}

export default function App() {
  useKeepAwake();

  const [serverUrl, setServerUrl] = useState("http://YOUR-PC-IP:8765");
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [messages, setMessages] = useState<Turn[]>([]);
  const [isRecording, setIsRecording] = useState(false);
  const [isBusy, setIsBusy] = useState(false);
  const [autoMode, setAutoMode] = useState(false);
  const [status, setStatus] = useState("Ready");

  const recordingRef = useRef<Audio.Recording | null>(null);
  const autoTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const soundRef = useRef<Audio.Sound | null>(null);
  const autoModeRef = useRef(false);
  const isBusyRef = useRef(false);
  const stopInFlightRef = useRef(false);
  const speechStartedRef = useRef(false);
  const silenceStartedAtRef = useRef<number | null>(null);
  const recordingStartedAtRef = useRef<number>(0);

  const canRecord = useMemo(
    () => !isBusy && !isRecording && serverUrl.includes("http"),
    [isBusy, isRecording, serverUrl]
  );

  useEffect(() => {
    autoModeRef.current = autoMode;
  }, [autoMode]);

  useEffect(() => {
    isBusyRef.current = isBusy;
  }, [isBusy]);

  async function ensureSessionId(): Promise<string> {
    if (sessionId) {
      return sessionId;
    }
    const resp = await fetch(`${normalizeBaseUrl(serverUrl)}/session/new`, {
      method: "POST",
    });
    const data = (await resp.json()) as { session_id: string };
    setSessionId(data.session_id);
    return data.session_id;
  }

  async function startRecording() {
    if (!canRecord) return;
    try {
      setStatus("Requesting microphone permission...");
      const perm = await Audio.requestPermissionsAsync();
      if (!perm.granted) {
        setStatus("Microphone permission denied");
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
      });

      speechStartedRef.current = false;
      silenceStartedAtRef.current = null;
      recordingStartedAtRef.current = Date.now();

      const rec = new Audio.Recording();
      const options = {
        ...Audio.RecordingOptionsPresets.HIGH_QUALITY,
        ios: {
          ...Audio.RecordingOptionsPresets.HIGH_QUALITY.ios,
          isMeteringEnabled: true,
        },
      } as Audio.RecordingOptions;
      await rec.prepareToRecordAsync(options);
      rec.setProgressUpdateInterval(200);
      rec.setOnRecordingStatusUpdate((recordingStatus) => {
        if (!autoModeRef.current || !recordingStatus.canRecord || stopInFlightRef.current) {
          return;
        }

        const now = Date.now();
        const metering =
          typeof recordingStatus.metering === "number"
            ? recordingStatus.metering
            : -160;

        if (metering > AUTO_SPEECH_START_DB) {
          speechStartedRef.current = true;
          silenceStartedAtRef.current = null;
        } else if (speechStartedRef.current) {
          if (!silenceStartedAtRef.current) {
            silenceStartedAtRef.current = now;
          } else if (now - silenceStartedAtRef.current >= AUTO_SILENCE_MS) {
            void stopAndSend(true);
            return;
          }
        }

        if (now - recordingStartedAtRef.current >= AUTO_MAX_RECORD_MS) {
          void stopAndSend(true);
        }
      });
      await rec.startAsync();
      recordingRef.current = rec;
      setIsRecording(true);
      setStatus(autoMode ? "Listening (auto mode)..." : "Listening...");
    } catch {
      setIsRecording(false);
      setStatus("Failed to start recording");
    }
  }

  async function stopAndSend(fromAuto = false) {
    const rec = recordingRef.current;
    if (!rec || stopInFlightRef.current) return;
    stopInFlightRef.current = true;

    try {
      setIsBusy(true);
      setIsRecording(false);
      setStatus("Processing...");
      await rec.stopAndUnloadAsync();
      const uri = rec.getURI();
      recordingRef.current = null;
      if (!uri) {
        setStatus("No audio captured");
        return;
      }

      const sid = await ensureSessionId();
      const form = new FormData();
      form.append("session_id", sid);
      form.append("speed", "1.2");
      form.append("audio", {
        uri,
        name: "utterance.m4a",
        type: "audio/m4a",
      } as any);

      const resp = await fetch(`${normalizeBaseUrl(serverUrl)}/talk`, {
        method: "POST",
        body: form,
      });

      const data = (await resp.json()) as TalkResponse;
      if (data.session_id && data.session_id !== sid) {
        setSessionId(data.session_id);
      }

      if (data.transcript) {
        setMessages((prev) => [...prev, { role: "user", text: data.transcript }]);
      }
      if (data.reply_text) {
        setMessages((prev) => [...prev, { role: "assistant", text: data.reply_text }]);
      }

      const wav = absoluteUrl(serverUrl, data.audio_url);
      if (wav) {
        setStatus("Playing response...");
        if (soundRef.current) {
          await soundRef.current.unloadAsync();
        }
        const { sound } = await Audio.Sound.createAsync({ uri: wav });
        soundRef.current = sound;
        await sound.playAsync();
        await new Promise<void>((resolve) => {
          sound.setOnPlaybackStatusUpdate((playback) => {
            if (!playback.isLoaded) return;
            if (playback.didJustFinish) {
              resolve();
            }
          });
        });
      }

      setStatus("Ready");
    } catch {
      setStatus("Failed to process turn");
    } finally {
      setIsBusy(false);
      stopInFlightRef.current = false;
      if (fromAuto && autoMode) {
        queueAutoCycle();
      }
    }
  }

  function clearAutoTimer() {
    if (autoTimerRef.current) {
      clearTimeout(autoTimerRef.current);
      autoTimerRef.current = null;
    }
  }

  function queueAutoCycle() {
    clearAutoTimer();
    autoTimerRef.current = setTimeout(async () => {
      if (!autoModeRef.current || isBusyRef.current || recordingRef.current) return;
      await startRecording();
    }, 250);
  }

  async function toggleAutoMode() {
    if (autoMode) {
      setAutoMode(false);
      clearAutoTimer();
      if (recordingRef.current) {
        await stopAndSend(false);
      } else {
        setStatus("Ready");
      }
      return;
    }

    setAutoMode(true);
    queueAutoCycle();
  }

  return (
    <LinearGradient colors={["#f2f7f5", "#e7eef9"]} style={styles.screen}>
      <StatusBar style="dark" />
      <View style={styles.header}>
        <Text style={styles.title}>OpenClaw Voice</Text>
        <Text style={styles.subtitle}>Simple mobile bridge for your local bot</Text>
      </View>

      <View style={styles.card}>
        <Text style={styles.label}>Server URL</Text>
        <TextInput
          style={styles.input}
          value={serverUrl}
          onChangeText={setServerUrl}
          autoCapitalize="none"
          autoCorrect={false}
          placeholder="http://192.168.x.x:8765"
        />
        <Text style={styles.status}>Status: {status}</Text>
      </View>

      <View style={styles.actions}>
        <Pressable
          style={[styles.button, !canRecord && styles.buttonDisabled]}
          onPress={startRecording}
          disabled={!canRecord}
        >
          <Text style={styles.buttonText}>Start</Text>
        </Pressable>
        <Pressable
          style={[styles.button, !isRecording && styles.buttonDisabled]}
          onPress={() => stopAndSend(false)}
          disabled={!isRecording}
        >
          <Text style={styles.buttonText}>Send</Text>
        </Pressable>
        <Pressable
          style={[styles.button, autoMode && styles.buttonActive]}
          onPress={toggleAutoMode}
        >
          <Text style={styles.buttonText}>{autoMode ? "Stop Auto" : "Auto Talk"}</Text>
        </Pressable>
      </View>

      {isBusy ? <ActivityIndicator size="small" color="#1f3a5f" /> : null}

      <ScrollView contentContainerStyle={styles.chat}>
        {messages.map((m, idx) => (
          <View
            key={`${m.role}-${idx}`}
            style={[styles.bubble, m.role === "user" ? styles.user : styles.bot]}
          >
            <Text style={styles.bubbleRole}>{m.role === "user" ? "You" : "OpenClaw"}</Text>
            <Text style={styles.bubbleText}>{m.text}</Text>
          </View>
        ))}
      </ScrollView>
    </LinearGradient>
  );
}

const styles = StyleSheet.create({
  screen: {
    flex: 1,
    paddingTop: 56,
    paddingHorizontal: 18,
  },
  header: {
    marginBottom: 16,
  },
  title: {
    fontSize: 32,
    color: "#11263f",
    fontWeight: "700",
    letterSpacing: -0.5,
  },
  subtitle: {
    color: "#395271",
    marginTop: 4,
    fontSize: 14,
  },
  card: {
    backgroundColor: "rgba(255,255,255,0.78)",
    borderRadius: 16,
    padding: 14,
    borderWidth: 1,
    borderColor: "#d6e0ef",
  },
  label: {
    fontSize: 12,
    color: "#2e4765",
    marginBottom: 6,
    fontWeight: "600",
  },
  input: {
    backgroundColor: "#fff",
    borderRadius: 10,
    borderColor: "#c7d5ea",
    borderWidth: 1,
    paddingHorizontal: 12,
    paddingVertical: 10,
    fontSize: 15,
    color: "#14304f",
  },
  status: {
    marginTop: 10,
    color: "#334f6d",
    fontSize: 13,
  },
  actions: {
    flexDirection: "row",
    justifyContent: "space-between",
    marginTop: 12,
    marginBottom: 12,
    gap: 8,
  },
  button: {
    flex: 1,
    backgroundColor: "#1f3a5f",
    paddingVertical: 12,
    borderRadius: 12,
    alignItems: "center",
  },
  buttonDisabled: {
    opacity: 0.45,
  },
  buttonActive: {
    backgroundColor: "#0f8a66",
  },
  buttonText: {
    color: "white",
    fontWeight: "700",
    fontSize: 14,
  },
  chat: {
    paddingBottom: 36,
    gap: 10,
  },
  bubble: {
    padding: 12,
    borderRadius: 14,
  },
  user: {
    backgroundColor: "#e4edf9",
  },
  bot: {
    backgroundColor: "#eef7ec",
  },
  bubbleRole: {
    fontSize: 12,
    fontWeight: "700",
    color: "#274463",
    marginBottom: 4,
  },
  bubbleText: {
    color: "#1a324f",
    lineHeight: 21,
    fontSize: 15,
  },
});
