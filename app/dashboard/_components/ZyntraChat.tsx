"use client";

import { useState, useRef, useEffect } from "react";
import type { ZyntraOutput } from "@/server/zyntra/types";
import type { ZyntraHackStatus } from "./ZyntraStatusCard";
import { speakText } from "../_lib/voiceAssistant";

interface Message {
  role: "user" | "zyntra";
  text: string;
  timestamp: Date;
}

const QUICK_ACTIONS = [
  { label: "How am I doing?", query: "How am I doing?" },
  { label: "Why?", query: "Why?" },
  { label: "What can I do?", query: "What can I do?" },
];

interface ZyntraChatProps {
  initialOutput: ZyntraOutput | ZyntraHackStatus | null;
  onClose: () => void;
}

export function ZyntraChat({ initialOutput, onClose }: ZyntraChatProps) {
  // Human-first initial message: use primaryMessage + coachMessage if available (ZyntraHackStatus)
  // otherwise fall back to the legacy ZyntraOutput shape
  const hackStatus = initialOutput as ZyntraHackStatus | null;
  const legacyOutput = initialOutput as ZyntraOutput | null;

  let initialMessage: string;
  if (hackStatus?.primaryMessage) {
    // Align chat opener precisely with the detected scenario
    const scenario = hackStatus.glucoseScenario ?? '';
    if (scenario === 'low') {
      initialMessage = `Your glucose may go low soon. You should eat something now. How else can I help?`;
    } else if (scenario === 'high') {
      initialMessage = `Your glucose may keep rising. A short walk could help bring it down. What would you like to know?`;
    } else if (scenario === 'unstable') {
      initialMessage = `Your glucose may become unstable. Try to keep your next meal balanced. Anything I can help with?`;
    } else {
      initialMessage = `Your glucose looks stable right now. Keep your current routine. Is there anything you'd like to check?`;
    }
  } else if (legacyOutput?.riskScore != null && legacyOutput.riskScore > 70) {
    initialMessage = `Your risk score is ${legacyOutput.riskScore}/100. ${legacyOutput.explanation ?? ""} What would you like to know?`;
  } else {
    initialMessage = "Hi! I'm Zyntra. Ask me anything about your current glucose status or what you should do right now.";
  }

  const [messages, setMessages] = useState<Message[]>([
    {
      role: "zyntra",
      text: initialMessage,
      timestamp: new Date(),
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  useEffect(() => {
    const SpeechRecognitionApi =
      typeof window !== "undefined"
        ? (window.SpeechRecognition || window.webkitSpeechRecognition)
        : undefined;

    if (!SpeechRecognitionApi) return;

    const recognition = new SpeechRecognitionApi();
    recognition.lang = "en-US";
    recognition.continuous = false;
    recognition.interimResults = false;

    recognition.onresult = (event: SpeechRecognitionEvent) => {
      const transcript = event.results?.[0]?.[0]?.transcript?.trim() ?? "";
      if (transcript) {
        setInput(transcript);
        void sendMessage(transcript);
      }
    };

    recognition.onend = () => setIsListening(false);
    recognition.onerror = () => setIsListening(false);
    recognitionRef.current = recognition;

    return () => {
      recognition.stop();
      recognitionRef.current = null;
    };
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function sendMessage(text: string) {
    const trimmedMessage = text.trim();
    if (!trimmedMessage || isLoading) return;

    const userMsg: Message = { role: "user", text: trimmedMessage, timestamp: new Date() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await fetch("/api/zyntra/conversation", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          message: trimmedMessage,
          // Pass current scenario context so the API can reply accordingly
          glucoseScenario: hackStatus?.glucoseScenario ?? null,
          primaryMessage: hackStatus?.primaryMessage ?? null,
          coachMessage: hackStatus?.coachMessage ?? null,
          explanation: hackStatus?.explanation ?? null,
          riskScore: hackStatus?.riskScore ?? null,
        }),
      });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(data?.error ?? "Conversation request failed");
      }
      setMessages((prev) => [
        ...prev,
        { role: "zyntra", text: data.reply ?? "I couldn't get a response. Please try again.", timestamp: new Date() },
      ]);
      if (data?.reply) {
        await speakText(data.reply, { preferElevenLabs: true });
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "zyntra", text: "Something went wrong. Please try again.", timestamp: new Date() },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  function handleKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  }

  function toggleVoiceInput() {
    const recognition = recognitionRef.current;
    if (!recognition || isLoading) return;

    if (isListening) {
      recognition.stop();
      setIsListening(false);
      return;
    }

    setIsListening(true);
    recognition.start();
  }

  return (
    <div
      id="zyntra-chat-overlay"
      className="fixed inset-0 z-[100] flex items-end justify-center bg-black/40 backdrop-blur-sm"
      onClick={(e) => { if (e.target === e.currentTarget) onClose(); }}
    >
      <div className="w-full max-w-lg bg-white rounded-t-[2rem] shadow-2xl flex flex-col"
           style={{ height: "82vh" }}>

        {/* Header */}
        <div className="flex items-center justify-between px-6 py-5 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-full bg-gradient-to-br from-zyntra-navy to-teal-600 flex items-center justify-center shadow-md">
              <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2}
                  d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" />
              </svg>
            </div>
            <div>
              <p className="font-bold text-slate-900 text-sm">Zyntra</p>
              <p className="text-xs text-teal-600 font-medium flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-teal-500 inline-block animate-pulse" />
                Active
              </p>
            </div>
          </div>
          <button
            id="zyntra-chat-close"
            onClick={onClose}
            className="w-8 h-8 rounded-full bg-slate-100 flex items-center justify-center text-slate-500 hover:bg-slate-200 transition-colors"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Message List */}
        <div className="flex-1 overflow-y-auto px-5 py-4 space-y-4">
          {messages.map((msg, i) => (
            <div key={i} className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}>
              {msg.role === "zyntra" && (
                <div className="w-7 h-7 rounded-full bg-gradient-to-br from-zyntra-navy to-teal-600 flex items-center justify-center text-white text-xs font-bold mr-2 mt-auto flex-shrink-0">
                  Z
                </div>
              )}
              <div
                className={`max-w-[78%] px-4 py-3 rounded-2xl text-sm leading-relaxed ${
                  msg.role === "user"
                    ? "bg-zyntra-navy text-white rounded-br-sm"
                    : "bg-slate-100 text-slate-800 rounded-bl-sm"
                }`}
              >
                {msg.text}
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="w-7 h-7 rounded-full bg-gradient-to-br from-zyntra-navy to-teal-600 flex items-center justify-center text-white text-xs font-bold mr-2 mt-auto flex-shrink-0">Z</div>
              <div className="bg-slate-100 rounded-2xl rounded-bl-sm px-4 py-3 flex gap-1.5 items-center">
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                <span className="w-2 h-2 bg-slate-400 rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Quick Actions */}
        <div className="px-5 pb-3 flex gap-2 overflow-x-auto no-scrollbar">
          {QUICK_ACTIONS.map((action) => (
            <button
              key={action.query}
              id={`zyntra-quick-${action.query.toLowerCase().replace(/[^a-z]/g, "-")}`}
              onClick={() => sendMessage(action.query)}
              disabled={isLoading}
              className="flex-shrink-0 px-4 py-2 border border-slate-200 rounded-full text-xs font-medium text-slate-700 hover:bg-slate-100 hover:border-slate-300 transition-all disabled:opacity-50 whitespace-nowrap"
            >
              {action.label}
            </button>
          ))}
        </div>

        {/* Input */}
        <div className="px-4 pb-6 pt-1">
          <div className="flex items-center gap-2 bg-slate-100 rounded-2xl px-4 py-3">
            <input
              id="zyntra-chat-input"
              type="text"
              placeholder="Ask Zyntra anything…"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              className="flex-1 bg-transparent text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none disabled:opacity-50"
            />
            <button
              id="zyntra-chat-mic"
              onClick={toggleVoiceInput}
              disabled={isLoading || !recognitionRef.current}
              className={`w-8 h-8 rounded-full flex items-center justify-center transition-all disabled:opacity-30 ${
                isListening ? "bg-rose-600 text-white animate-pulse" : "bg-slate-200 text-slate-700 hover:bg-slate-300"
              }`}
              title={recognitionRef.current ? "Speak to Zyntra" : "Speech recognition not available in this browser"}
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 1v11m0 0a3 3 0 003-3V5a3 3 0 10-6 0v4a3 3 0 003 3zm-7 0a7 7 0 0014 0M8 21h8"
                />
              </svg>
            </button>
            <button
              id="zyntra-chat-send"
              onClick={() => sendMessage(input)}
              disabled={!input.trim() || isLoading}
              className="w-8 h-8 rounded-full bg-zyntra-navy flex items-center justify-center text-white transition-all disabled:opacity-30 hover:bg-slate-800"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
              </svg>
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

type SpeechRecognitionEvent = Event & {
  results: SpeechRecognitionResultList;
};

type SpeechRecognition = {
  lang: string;
  continuous: boolean;
  interimResults: boolean;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onend: (() => void) | null;
  onerror: (() => void) | null;
  start: () => void;
  stop: () => void;
};

type SpeechRecognitionConstructor = new () => SpeechRecognition;

declare global {
  interface Window {
    SpeechRecognition?: SpeechRecognitionConstructor;
    webkitSpeechRecognition?: SpeechRecognitionConstructor;
  }
}
