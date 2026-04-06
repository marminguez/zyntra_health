"use client";

interface SpeakOptions {
  preferElevenLabs?: boolean;
  voiceId?: string;
}

export async function speakText(text: string, options: SpeakOptions = {}): Promise<void> {
  const trimmedText = text.trim();
  if (!trimmedText) return;

  if (options.preferElevenLabs) {
    const spokenByElevenLabs = await trySpeakWithElevenLabs(trimmedText, options.voiceId);
    if (spokenByElevenLabs) return;
  }

  speakWithBrowserVoice(trimmedText);
}

async function trySpeakWithElevenLabs(text: string, voiceId?: string): Promise<boolean> {
  try {
    const response = await fetch("/api/zyntra/voice", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, voiceId }),
    });

    if (!response.ok) return false;

    const blob = await response.blob();
    const audioUrl = URL.createObjectURL(blob);
    const audio = new Audio(audioUrl);
    audio.onended = () => URL.revokeObjectURL(audioUrl);
    await audio.play();
    return true;
  } catch {
    return false;
  }
}

function speakWithBrowserVoice(text: string): void {
  if (typeof window === "undefined" || !("speechSynthesis" in window)) return;

  const utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = "en-US";
  utterance.rate = 1;
  utterance.pitch = 1.1;

  const voices = window.speechSynthesis.getVoices();
  const preferredVoice = voices.find((voice) =>
    /(female|samantha|victoria|zira|aria|serena|jenny|ava)/i.test(voice.name)
  );
  if (preferredVoice) utterance.voice = preferredVoice;

  window.speechSynthesis.cancel();
  window.speechSynthesis.speak(utterance);
}
