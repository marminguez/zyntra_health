"use client";

import { useEffect, useState, useRef } from "react";
import { speakText } from "../_lib/voiceAssistant";

interface ZyntraAlertOverlayProps {
  payload: {
    alertTitle: string;
    alertMessage: string;
    lockscreenPreview: string;
    voiceText: string;
    alertSeverity: string;
  };
  onOpenApp: () => void;
}

export function ZyntraAlertOverlay({ payload, onOpenApp }: ZyntraAlertOverlayProps) {
  const [isVisible, setIsVisible] = useState(false);
  const [time, setTime] = useState("");
  const [date, setDate] = useState("");
  const [hasSpoken, setHasSpoken] = useState(false);
  const voiceTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    setIsVisible(true);

    // Format time and date
    const now = new Date();
    setTime(now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", hour12: false }));
    setDate(now.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" }));

    // Auto-play voice with 400ms delay — natural, not jarring
    if (!hasSpoken) {
      voiceTimerRef.current = setTimeout(() => {
        void speakText(payload.voiceText, { preferElevenLabs: true });
        setHasSpoken(true);
      }, 400);
    }

    return () => {
      if (voiceTimerRef.current) clearTimeout(voiceTimerRef.current);
    };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const isHighGlucose = payload.alertSeverity !== "low" && payload.alertTitle.toLowerCase().includes("high");
  const accentColor = isHighGlucose ? "bg-red-500" : "bg-orange-500";

  return (
    <div
      className={`fixed inset-0 z-[100] flex flex-col items-center justify-between py-20 px-6 transition-all duration-700 ${
        isVisible ? "opacity-100" : "opacity-0 pointer-events-none"
      }`}
      style={{ background: "linear-gradient(180deg, #060D17 0%, #0B1929 60%, #0F2236 100%)" }}
    >
      {/* ── Lock screen clock ── */}
      <div className="flex flex-col items-center animate-in fade-in zoom-in-95 duration-1000">
        <p className="text-white/50 text-base font-medium tracking-wide mb-2">{date}</p>
        <p className="text-white text-8xl font-extralight tracking-tighter leading-none">{time}</p>
      </div>

      {/* ── Notification card ── */}
      <div className="w-full max-w-sm animate-in slide-in-from-bottom-12 duration-500 delay-200">
        
        {/* Accent bar */}
        <div className={`h-1 w-16 rounded-full ${accentColor} mb-4 mx-auto`} />

        <div
          className="bg-white/[0.08] border border-white/[0.12] backdrop-blur-2xl rounded-3xl p-5 shadow-[0_32px_80px_rgba(0,0,0,0.6)]"
        >
          {/* App header row */}
          <div className="flex items-center justify-between mb-3">
            <div className="flex items-center gap-2">
              <div
                style={{ backgroundColor: "#08253F" }}
                className="w-7 h-7 rounded-xl flex items-center justify-center shadow-lg"
              >
                <svg className="w-3.5 h-3.5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2.5}
                    d="M13 10V3L4 14h7v7l9-11h-7z" />
                </svg>
              </div>
              <span className="text-white/60 text-xs font-bold uppercase tracking-widest">Zyntra</span>
            </div>
            <span className="text-white/30 text-[11px] font-medium">now</span>
          </div>

          {/* Alert content */}
          <h3 className="text-white font-black text-xl leading-tight mb-2">
            {payload.alertTitle}
          </h3>
          <p className="text-white/70 text-[15px] leading-snug">
            {payload.alertMessage}
          </p>
        </div>
      </div>

      {/* ── Open App button ── */}
      <div className="flex flex-col items-center gap-4 animate-in fade-in duration-1000 delay-500">
        <button
          id="zyntra-alert-open-app"
          onClick={onOpenApp}
          style={{ backgroundColor: "#08253F" }}
          className="px-8 py-4 rounded-2xl text-white text-sm font-black uppercase tracking-widest shadow-xl hover:opacity-90 active:scale-95 transition-all border border-white/10"
        >
          Open Zyntra →
        </button>
        <p className="text-white/30 text-xs font-medium tracking-widest uppercase">
          Tap to view your status
        </p>
      </div>
    </div>
  );
}
