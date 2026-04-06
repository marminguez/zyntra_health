"use client";

import { useSession, signOut } from "next-auth/react";
import { useState, useEffect, useRef } from "react";
import { useSearchParams } from "next/navigation";
import type { MLPayload } from "./ClinicianDashboard";
import type { ZyntraOutput } from "@/server/zyntra/types";

import { currentRiskProfile } from "../_lib/mockRiskData";
import { DashboardView } from "./views/DashboardView";
import { DetailedPredictionView } from "./views/DetailedPredictionView";
import { BaselineView } from "./views/BaselineView";
import { RecommendationsView } from "./views/RecommendationsView";
import { HistoryView } from "./views/HistoryView";
import { ZyntraChat } from "./ZyntraChat";
import { ZyntraStatusCard, ZyntraHackStatus } from "./ZyntraStatusCard";
import { ZyntraAlertOverlay } from "./ZyntraAlertOverlay";
import { speakText } from "../_lib/voiceAssistant";

const ZYNTRA_ALERT_THRESHOLD = 70;
const ZYNTRA_PREVENTIVE_THRESHOLD = 60;
type LibreWizardStatus =
    | "NOT_STARTED"
    | "INVITE_SENT"
    | "WAITING_FOR_LIBRELINKUP_ACCEPTANCE"
    | "SHARE_ACCEPTED_NO_DATA_YET"
    | "WAITING_FOR_DATA"
    | "SYNC_ACTIVE"
    | "SYNC_ERROR"
    | "EMAIL_MISMATCH"
    | "NETWORK_OR_UPLOAD_DELAY";

export function PatientDashboard() {
    const { data: session } = useSession();
    const patientId = (session?.user as any)?.patientId ?? (session?.user as any)?.id;
    // Keep existing payload for potential background updates
    const [payload, setPayload] = useState<MLPayload | null>(null);
    const [loading, setLoading] = useState(false);
    
    // Zyntra state
    const [zyntraData, setZyntraData] = useState<ZyntraHackStatus | null>(null);
    const [isChatOpen, setIsChatOpen] = useState(false);
    const [zyntraLoading, setZyntraLoading] = useState(false);
    const [voiceAlertsEnabled, setVoiceAlertsEnabled] = useState(true);
    const [screenOffSimulationEnabled, setScreenOffSimulationEnabled] = useState(false);
    const [showDemoAlert, setShowDemoAlert] = useState(false);
    const statusCardRef = useRef<HTMLDivElement>(null);
    const hasAnnouncedRiskRef = useRef(false);
    const hasAnnouncedPreventiveRef = useRef(false);

    // New navigation state
    const [activeTab, setActiveTab] = useState<"STATUS" | "INSIGHTS" | "BASELINE" | "RECOMMENDATIONS" | "HISTORY" | "DEVICES">("STATUS");
    const [drilldown, setDrilldown] = useState<"48h" | "72h" | null>(null);
    const [scenario, setScenario] = useState<string>("latest");
    
    // Devices logic
    const searchParams = useSearchParams();
    const [syncMessage, setSyncMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);
    const [isSyncingFitbit, setIsSyncingFitbit] = useState(false);
    const [isSyncingLibre, setIsSyncingLibre] = useState(false);
    const [isConnectingLibre, setIsConnectingLibre] = useState(false);
    const [isAcceptingLibre, setIsAcceptingLibre] = useState(false);
    const [isResettingLibre, setIsResettingLibre] = useState(false);
    const [libreEmail, setLibreEmail] = useState("");
    const [librePassword, setLibrePassword] = useState("");
    const [libreAcceptedEmail, setLibreAcceptedEmail] = useState("");
    const [libreStatus, setLibreStatus] = useState<LibreWizardStatus>("NOT_STARTED");
    const [libreLastSyncAt, setLibreLastSyncAt] = useState<string | null>(null);
    const [libreAcceptedAt, setLibreAcceptedAt] = useState<string | null>(null);
    const [libreLastDataAt, setLibreLastDataAt] = useState<string | null>(null);
    const [libreErrorMessage, setLibreErrorMessage] = useState<string | null>(null);
    const [showLibreAdvanced, setShowLibreAdvanced] = useState(false);

    useEffect(() => {
        if (searchParams?.get("fitbit") === "connected") {
            setActiveTab("DEVICES");
            setSyncMessage({ type: "success", text: "Fitbit connected successfully!" });
        }
        if (searchParams?.get("fitbit") === "error") {
            setActiveTab("DEVICES");
            setSyncMessage({ type: "error", text: decodeURIComponent(searchParams.get("message") ?? "Fitbit connection failed") });
        }
    }, [searchParams]);

    useEffect(() => {
        const fetchLibreStatus = async () => {
            if (!patientId) return;
            try {
                const res = await fetch(`/app-api/integrations/freestyle/status?patientId=${encodeURIComponent(patientId)}`, {
                    credentials: "include",
                });
                if (!res.ok) return;
                const data = await res.json();
                setLibreStatus((data.status as LibreWizardStatus) ?? "NOT_STARTED");
                setLibreLastSyncAt(typeof data?.connection?.lastCheckAt === "string" ? data.connection.lastCheckAt : null);
                setLibreAcceptedAt(typeof data?.connection?.acceptedAt === "string" ? data.connection.acceptedAt : null);
                setLibreLastDataAt(typeof data?.connection?.lastDataAt === "string" ? data.connection.lastDataAt : null);
                setLibreErrorMessage(typeof data?.connection?.errorMessage === "string" ? data.connection.errorMessage : null);
                if (typeof data?.connection?.invitedEmail === "string") {
                    setLibreEmail(data.connection.invitedEmail.trim().toLowerCase() || "");
                }
                if (typeof data?.connection?.acceptedEmail === "string") setLibreAcceptedEmail(data.connection.acceptedEmail);
            } catch (err) {
                console.error("Failed to load LibreLink status", err);
            }
        };

        if (activeTab === "DEVICES") {
            void fetchLibreStatus();
        }
    }, [activeTab, patientId]);

    useEffect(() => {
        const shouldAutoCheck =
            activeTab === "DEVICES" &&
            (libreStatus === "WAITING_FOR_DATA" ||
                libreStatus === "SHARE_ACCEPTED_NO_DATA_YET" ||
                libreStatus === "NETWORK_OR_UPLOAD_DELAY");

        if (!shouldAutoCheck || !patientId) return;

        const interval = setInterval(() => {
            void fetch("/app-api/integrations/freestyle/sync", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({
                    patientId,
                    isOnline: typeof navigator !== "undefined" ? navigator.onLine : true,
                    region: Intl.DateTimeFormat().resolvedOptions().timeZone,
                }),
            });
        }, 60000);

        return () => clearInterval(interval);
    }, [activeTab, libreStatus, patientId]);

    useEffect(() => {
        const fetchMyRisk = async () => {
            if (!patientId) return;
            setLoading(true);
            try {
                const res = await fetch("/app-api/risk", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ patientId }),
                });

                if (res.ok) {
                    const data = await res.json();
                    setPayload(data);
                }
            } catch (err) {
                console.error("Failed to fetch risk", err);
            } finally {
                setLoading(false);
            }
        };

        if (patientId && activeTab === "STATUS") {
            fetchMyRisk();
        }
    }, [patientId, activeTab]);

    // Fetch Zyntra status on dashboard mount
    async function fetchZyntraStatus() {
        setZyntraLoading(true);
        try {
            // Updated to ZyntraHack endpoint for Participant 2302 with scenario support
            const res = await fetch(`/api/zyntrahack/status?participantId=2302&scenario=${scenario}`);
            if (res.ok) {
                const data = await res.json();
                setZyntraData(data);
            }
        } catch (err) {
            console.error("Failed to fetch Zyntra status", err);
        } finally {
            setZyntraLoading(false);
        }
    }

    useEffect(() => {
        fetchZyntraStatus();
    }, [scenario]);

    useEffect(() => {
        if (activeTab === "STATUS" || activeTab === "INSIGHTS") {
            void fetchZyntraStatus();
        }
    }, [activeTab]);

    useEffect(() => {
        if (!voiceAlertsEnabled || !zyntraData) return;

        if (zyntraData.riskScore > ZYNTRA_ALERT_THRESHOLD && !hasAnnouncedRiskRef.current) {
            hasAnnouncedRiskRef.current = true;
            // Use the scenario-specific voice message — never the generic fallback
            void speakText(
                zyntraData.voicePayload?.voiceText ?? `Zyntra alert. ${zyntraData.primaryMessage}. ${zyntraData.coachMessage}`,
                { preferElevenLabs: true }
            );
            return;
        }

        if (zyntraData.riskScore <= ZYNTRA_ALERT_THRESHOLD) {
            hasAnnouncedRiskRef.current = false;
        }
    }, [voiceAlertsEnabled, zyntraData]);

    useEffect(() => {
        if (!voiceAlertsEnabled || !screenOffSimulationEnabled || !zyntraData) return;

        // Use the scenario voice text for preventive alerts too
        const preventiveMessage = zyntraData.voicePayload?.voiceText
            ?? `Preventive alert. ${zyntraData.primaryMessage}. ${zyntraData.coachMessage}`;

        const shouldTriggerPreventiveAlert =
            zyntraData.riskScore >= ZYNTRA_PREVENTIVE_THRESHOLD &&
            zyntraData.riskScore <= ZYNTRA_ALERT_THRESHOLD;

        if (shouldTriggerPreventiveAlert && !hasAnnouncedPreventiveRef.current) {
            hasAnnouncedPreventiveRef.current = true;
            void speakText(preventiveMessage, { preferElevenLabs: true });

            if ("Notification" in window && Notification.permission === "granted") {
                new Notification("Zyntra", {
                    body: zyntraData.alertPayload?.alertMessage ?? zyntraData.primaryMessage,
                });
            }
            return;
        }

        if (!shouldTriggerPreventiveAlert) {
            hasAnnouncedPreventiveRef.current = false;
        }
    }, [voiceAlertsEnabled, screenOffSimulationEnabled, zyntraData]);

    useEffect(() => {
        if (!voiceAlertsEnabled || !screenOffSimulationEnabled || !zyntraData) return;

        const onVisibilityChange = () => {
            const isHidden = document.visibilityState === "hidden";
            const isRisingRisk =
                zyntraData.riskScore >= ZYNTRA_PREVENTIVE_THRESHOLD &&
                zyntraData.riskScore <= ZYNTRA_ALERT_THRESHOLD;

            if (!isHidden || !isRisingRisk) return;

            void speakText(
                zyntraData.voicePayload?.voiceText
                    ?? `Zyntra. ${zyntraData.primaryMessage}. ${zyntraData.coachMessage}`,
                { preferElevenLabs: true }
            );

            if ("Notification" in window && Notification.permission === "granted") {
                new Notification("Zyntra screen-off simulation", {
                    body: "Preventive action now: hydrate, avoid heavy carbs, and walk 10 minutes.",
                });
            }
        };

        document.addEventListener("visibilitychange", onVisibilityChange);
        return () => document.removeEventListener("visibilitychange", onVisibilityChange);
    }, [voiceAlertsEnabled, screenOffSimulationEnabled, zyntraData]);

    async function toggleScreenOffSimulation() {
        if (!screenOffSimulationEnabled && "Notification" in window && Notification.permission === "default") {
            await Notification.requestPermission();
        }
        setScreenOffSimulationEnabled((prev) => !prev);
    }

    async function handleSyncFitbit() {
        setIsSyncingFitbit(true);
        setSyncMessage(null);
        try {
            const res = await fetch("/app-api/integrations/fitbit/sync", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ patientId }),
            });
            const data = await res.json();
            if (res.ok) {
                setSyncMessage({ type: "success", text: "Fitbit data synced successfully" });
            } else {
                setSyncMessage({ type: "error", text: `Sync failed: ${data.error || "Unknown error"}` });
            }
        } catch (err) {
            setSyncMessage({ type: "error", text: "Sync failed: Network error" });
        } finally {
            setIsSyncingFitbit(false);
        }
    }

    async function handleSyncLibre() {
        setIsSyncingLibre(true);
        setSyncMessage(null);
        try {
            const res = await fetch("/app-api/integrations/freestyle/sync", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({
                    patientId,
                    isOnline: typeof navigator !== "undefined" ? navigator.onLine : true,
                    region: Intl.DateTimeFormat().resolvedOptions().timeZone,
                }),
            });
            const data = await res.json();
            if (res.ok) {
                setLibreStatus((data.status as LibreWizardStatus) ?? "NOT_STARTED");
                setLibreLastSyncAt(typeof data.lastCheckAt === "string" ? data.lastCheckAt : new Date().toISOString());
                setLibreLastDataAt(typeof data.lastDataAt === "string" ? data.lastDataAt : null);
                setLibreAcceptedAt(typeof data.acceptedAt === "string" ? data.acceptedAt : libreAcceptedAt);
                if (data.errorMessage) {
                    setSyncMessage({ type: "error", text: data.errorMessage });
                } else {
                    setSyncMessage({ type: "success", text: "Libre connection verified and data is available." });
                }
                await fetchZyntraStatus();
            } else {
                setLibreStatus("SYNC_ERROR");
                setSyncMessage({ type: "error", text: `Connection check failed: ${data.error || "Unknown error"}` });
            }
        } catch (err) {
            setLibreStatus("NETWORK_OR_UPLOAD_DELAY");
            setSyncMessage({ type: "error", text: "Connection check could not finish due to network delay." });
        } finally {
            setIsSyncingLibre(false);
        }
    }

    async function handleConnectLibre() {
        if (!libreEmail.trim()) {
            setSyncMessage({ type: "error", text: "Enter your LibreLinkUp email." });
            return;
        }
        if (!librePassword.trim()) {
            setSyncMessage({ type: "error", text: "Enter your LibreLinkUp password." });
            return;
        }

        setIsConnectingLibre(true);
        setSyncMessage(null);
        try {
            const res = await fetch("/app-api/integrations/freestyle/connect", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({
                    patientId,
                    invitedEmail: libreEmail.trim(),
                    librePassword,
                    region: Intl.DateTimeFormat().resolvedOptions().timeZone,
                }),
            });
            const data = await res.json();
            if (res.ok) {
                setLibreStatus((data.status as LibreWizardStatus) ?? "WAITING_FOR_DATA");
                setLibreAcceptedAt(typeof data.acceptedAt === "string" ? data.acceptedAt : libreAcceptedAt);
                setSyncMessage({
                    type: "success",
                    text:
                        data.mode === "DIRECT_LOGIN"
                            ? "Libre credentials validated. Connection saved for direct sync."
                            : "Invitation registered. Accept it in LibreLinkUp with the same email.",
                });
            } else {
                setSyncMessage({ type: "error", text: `Connection failed: ${data.error || "Unknown error"}` });
            }
        } catch {
            setSyncMessage({ type: "error", text: "Connection failed: Network error" });
        } finally {
            setIsConnectingLibre(false);
        }
    }

    async function handleConfirmLibreAcceptance() {
        if (!libreAcceptedEmail.trim()) {
            setSyncMessage({ type: "error", text: "Enter the email used to accept inside LibreLinkUp." });
            return;
        }
        setIsAcceptingLibre(true);
        setSyncMessage(null);
        try {
            const res = await fetch("/app-api/integrations/freestyle/accept", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ patientId, acceptedEmail: libreAcceptedEmail.trim() }),
            });
            const data = await res.json();
            if (res.ok) {
                setLibreStatus((data.status as LibreWizardStatus) ?? "SHARE_ACCEPTED_NO_DATA_YET");
                setLibreAcceptedAt(typeof data.acceptedAt === "string" ? data.acceptedAt : new Date().toISOString());
                setSyncMessage({ type: "success", text: "Acceptance captured. Now keep Libre phone online and run Check connection." });
            } else {
                setSyncMessage({ type: "error", text: data.error || "Could not confirm invitation acceptance." });
            }
        } finally {
            setIsAcceptingLibre(false);
        }
    }

    async function handleResetLibre() {
        setIsResettingLibre(true);
        try {
            await fetch("/app-api/integrations/freestyle/reset", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ patientId }),
            });
            setLibreStatus("NOT_STARTED");
            setLibreEmail("");
            setLibrePassword("");
            setLibreAcceptedEmail("");
            setLibreAcceptedAt(null);
            setLibreLastDataAt(null);
            setLibreLastSyncAt(null);
            setLibreErrorMessage(null);
            setShowLibreAdvanced(false);
            setSyncMessage({ type: "success", text: "Libre onboarding reset. Start again from Step 1." });
        } finally {
            setIsResettingLibre(false);
        }
    }

    return (
        <>
        <div className="min-h-screen bg-slate-50 text-slate-900 pb-28">
            <header className="px-5 py-4 flex items-center justify-between border-b border-slate-100 bg-white/80 backdrop-blur-sm sticky top-0 z-40">
                <div className="flex items-center gap-2 cursor-pointer" onClick={() => signOut({ callbackUrl: "/login" })}>
                    {/* Zyntra Logo — uses the real SVG from /public */}
                    <img src="/zyntra-logo.svg" alt="Zyntra" className="h-7 w-auto" />
                </div>
                <div className="flex items-center gap-2">
                    <div className="w-8 h-8 rounded-full bg-slate-200 overflow-hidden relative shadow-sm border border-slate-300">
                    <svg className="w-8 h-8 text-slate-400 absolute bottom-0 translate-y-1" fill="currentColor" viewBox="0 0 24 24">
                        <path d="M24 20.993V24H0v-2.996A14.977 14.977 0 0112.004 15c4.904 0 9.26 2.354 11.996 5.993zM16.002 8.999a4 4 0 11-8 0 4 4 0 018 0z" />
                    </svg>
                    </div>
                </div>
            </header>

            {/* Removed redundant alert banners — info is already in the status card */}

            <div className="px-6">
                {activeTab === "STATUS" && (
                    <div className="max-w-xl mx-auto pt-6 pb-4">
                        {/* ZyntraStatusCard */}
                        {zyntraData && (
                            <div ref={statusCardRef} className="scroll-mt-6">
                                <ZyntraStatusCard
                                    output={zyntraData}
                                    loading={zyntraLoading}
                                    onTalkToZyntra={() => setIsChatOpen(true)}
                                    onSpeak={(txt) => void speakText(txt, { preferElevenLabs: true })}
                                />
                            </div>
                        )}
                        {zyntraLoading && !zyntraData && (
                            <div className="mb-8 bg-white rounded-[2.5rem] p-8 shadow-sm animate-pulse">
                                <div className="h-10 bg-slate-100 rounded-xl w-3/4 mb-4" />
                                <div className="h-6 bg-slate-100 rounded-lg w-1/2 mb-8" />
                                <div className="h-32 bg-slate-100 rounded-3xl w-full" />
                            </div>
                        )}
                    </div>
                )}
                {activeTab === "INSIGHTS" && (
                    <div className="animate-in fade-in duration-500">
                        <h1 className="text-3xl font-serif font-bold text-zyntra-navy mb-2">Metabolic Insights</h1>
                        <p className="text-slate-500 mb-8">Long-term patterns and predictive outlooks.</p>
                        {!drilldown && <DashboardView data={currentRiskProfile} onDrilldown={(t) => setDrilldown(t)} />}
                        {drilldown && <DetailedPredictionView data={currentRiskProfile} timeframe={drilldown} onBack={() => setDrilldown(null)} />}
                    </div>
                )}
                {activeTab === "BASELINE" && <BaselineView glucoseScenario={zyntraData?.glucoseScenario ?? "stable"} />}
                {activeTab === "RECOMMENDATIONS" && <RecommendationsView glucoseScenario={zyntraData?.glucoseScenario ?? "stable"} />}
                {activeTab === "HISTORY" && <HistoryView glucoseScenario={zyntraData?.glucoseScenario ?? "stable"} />}
                {activeTab === "DEVICES" && (
                    <div className="animate-in fade-in duration-300">
                        <h1 className="text-3xl font-serif font-bold text-zyntra-navy mb-2">Devices</h1>
                        <p className="text-slate-500 mb-8">Connect your wearables and medical devices to sync data with Zyntra Models.</p>

                        {syncMessage && (
                            <div className={`mb-6 p-4 rounded-xl text-sm font-medium ${syncMessage.type === "success" ? "bg-green-50 text-green-800 border border-green-200" : "bg-rose-50 text-rose-800 border border-rose-200"}`}>
                                {syncMessage.text}
                            </div>
                        )}

                        <div className="space-y-4">
                            {/* Fitbit Card */}
                            <div className="bg-white border border-slate-100 p-6 rounded-[1.5rem] shadow-sm flex flex-col gap-4">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 bg-teal-50 rounded-full flex items-center justify-center text-teal-600 font-bold text-xl">F</div>
                                        <div>
                                            <h3 className="font-bold text-lg text-slate-900">Fitbit</h3>
                                            <p className="text-sm text-slate-500">Activity, Sleep, HR</p>
                                        </div>
                                    </div>
                                    <span className="px-3 py-1 bg-green-50 text-green-700 text-xs font-bold rounded-full border border-green-100">Active</span>
                                </div>
                                <div className="flex gap-3 mt-2">
                                    <a href="/app-api/integrations/fitbit/connect" className="flex-1 bg-zyntra-navy text-white text-center py-3 rounded-xl font-medium text-sm transition-colors hover:bg-slate-800">
                                        Reconnect
                                    </a>
                                    <button onClick={handleSyncFitbit} disabled={isSyncingFitbit} className="flex-1 border border-slate-200 text-slate-700 text-center py-3 rounded-xl font-medium text-sm transition-colors hover:bg-slate-50 disabled:opacity-50">
                                        {isSyncingFitbit ? "Syncing..." : "Sync Now"}
                                    </button>
                                </div>
                            </div>

                            {/* Libre Card */}
                            <div className="bg-white border border-slate-100 p-6 rounded-[1.5rem] shadow-sm flex flex-col gap-4">
                                <div className="flex items-center justify-between">
                                    <div className="flex items-center gap-3">
                                        <div className="w-12 h-12 bg-amber-50 rounded-full flex items-center justify-center text-amber-600 font-bold text-xl">L</div>
                                        <div>
                                            <h3 className="font-bold text-lg text-slate-900">FreeStyle Libre</h3>
                                            <p className="text-sm text-slate-500">Continuous Glucose</p>
                                        </div>
                                    </div>
                                    <span className="px-3 py-1 bg-green-50 text-green-700 text-xs font-bold rounded-full border border-green-100">Sync active</span>
                                </div>
                                {libreLastSyncAt && (
                                    <p className="text-xs text-slate-500 -mt-2">
                                        Last check: {new Date(libreLastSyncAt).toLocaleString()}
                                    </p>
                                )}
                                {libreAcceptedAt && <p className="text-xs text-slate-500 -mt-2">Invitation accepted at: {new Date(libreAcceptedAt).toLocaleString()}</p>}
                                {libreLastDataAt && <p className="text-xs text-slate-500 -mt-2">Last glucose timestamp: {new Date(libreLastDataAt).toLocaleString()}</p>}
                                {libreErrorMessage && <p className="text-xs text-amber-700 bg-amber-50 border border-amber-200 rounded-lg p-2">{libreErrorMessage}</p>}
                                {(libreStatus === "SHARE_ACCEPTED_NO_DATA_YET" || libreStatus === "WAITING_FOR_DATA" || libreStatus === "SYNC_ACTIVE") && !showLibreAdvanced ? (
                                    <div className="rounded-xl border border-teal-100 bg-teal-50 p-3 text-sm text-teal-900">
                                        <p className="font-semibold">Follower share already linked.</p>
                                        <p className="mt-1 text-teal-800">
                                            Zyntra will auto-check every minute and activate sync when cloud glucose is available.
                                        </p>
                                        <button
                                            onClick={() => setShowLibreAdvanced(true)}
                                            className="mt-3 text-xs font-semibold text-teal-700 underline"
                                        >
                                            Edit invitation/acceptance details
                                        </button>
                                    </div>
                                ) : (
                                    <>
                                        <ol className="text-sm text-slate-600 list-decimal pl-5 space-y-1">
                                            <li>Enter your LibreLinkUp email and password below.</li>
                                            <li>Zyntra validates those credentials against LibreLinkUp.</li>
                                            <li>Keep your Libre app online so glucose uploads to the cloud.</li>
                                            <li>Tap Check connection to pull new glucose readings.</li>
                                        </ol>
                                        <div className="grid grid-cols-1 gap-3">
                                            <input
                                                type="email"
                                                value={libreEmail}
                                                onChange={(e) => setLibreEmail(e.target.value)}
                                                placeholder="Step 2: invited email"
                                                className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                                            />
                                            <input
                                                type="password"
                                                value={librePassword}
                                                onChange={(e) => setLibrePassword(e.target.value)}
                                                placeholder="Step 2: LibreLinkUp password"
                                                className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                                            />
                                            <input
                                                type="email"
                                                value={libreAcceptedEmail}
                                                onChange={(e) => setLibreAcceptedEmail(e.target.value)}
                                                placeholder="Step 3: accepted email in LibreLinkUp"
                                                className="w-full border border-slate-200 rounded-xl px-3 py-2.5 text-sm text-slate-800 placeholder:text-slate-400 focus:outline-none focus:ring-2 focus:ring-amber-200"
                                            />
                                        </div>
                                        <p className="text-xs text-slate-500">
                                            These credentials must match the LibreLinkUp app account used by the patient.
                                        </p>
                                        <div className="flex gap-3 mt-2">
                                            <button onClick={handleConnectLibre} disabled={isConnectingLibre} className="flex-1 border border-amber-200 text-amber-700 text-center py-3 rounded-xl font-medium text-sm transition-colors hover:bg-amber-50 disabled:opacity-50">
                                                {isConnectingLibre ? "Saving..." : "Save Libre credentials"}
                                            </button>
                                            <button onClick={handleConfirmLibreAcceptance} disabled={isAcceptingLibre} className="flex-1 border border-slate-200 text-slate-700 text-center py-3 rounded-xl font-medium text-sm transition-colors hover:bg-slate-50 disabled:opacity-50">
                                                {isAcceptingLibre ? "Confirming..." : "I accepted invitation"}
                                            </button>
                                        </div>
                                    </>
                                )}
                                <p className="text-sm text-slate-600">
                                    Conectado vía global `.env`. Email: <span className="font-semibold">marminguez@yahoo.es</span>
                                </p>
                                <div className="flex gap-3 mt-2">
                                    <button onClick={handleSyncLibre} disabled={isSyncingLibre} className="flex-1 bg-amber-600 text-white text-center py-3 rounded-xl font-medium text-sm transition-colors hover:bg-amber-700 disabled:opacity-50">
                                        {isSyncingLibre ? "Syncing..." : "Sync Now"}
                                    </button>
                                </div>
                            </div>
                        </div>
                    </div>
                )}
            </div>

            {/* Bottom Nav */}
            <div className="fixed bottom-0 left-0 right-0 bg-white border-t border-slate-100 px-6 pt-4 pb-6 flex justify-between items-center z-50 rounded-t-3xl shadow-[0_-10px_40px_rgba(0,0,0,0.03)]">
                <NavIcon 
                    icon={<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 12l2-2m0 0l7-7 7 7M5 10v10a1 1 0 001 1h3m10-11l2 2m-2-2v10a1 1 0 01-1 1h-3m-6 0a1 1 0 001-1v-4a1 1 0 011-1h2a1 1 0 011 1v4a1 1 0 001 1m-6 0h6" />}
                    label="HOME" active={activeTab === "STATUS"} 
                    onClick={() => { setActiveTab("STATUS"); setDrilldown(null); }} 
                />
                <NavIcon 
                    icon={<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />}
                    label="INSIGHTS" active={activeTab === "INSIGHTS"} 
                    onClick={() => { setActiveTab("INSIGHTS"); setDrilldown(null); }} 
                />
                <NavIcon 
                    icon={<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />}
                    label="BASELINE" active={activeTab === "BASELINE"} 
                    onClick={() => setActiveTab("BASELINE")} 
                />
                <NavIcon 
                    icon={<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2m-6 9l2 2 4-4" />}
                    label="ACTIONS" active={activeTab === "RECOMMENDATIONS"} 
                    onClick={() => setActiveTab("RECOMMENDATIONS")} 
                />
                <NavIcon 
                    icon={<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z" />}
                    label="HISTORY" active={activeTab === "HISTORY"} 
                    onClick={() => setActiveTab("HISTORY")} 
                />
                <NavIcon 
                    icon={<path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 18h.01M8 21h8a2 2 0 002-2V5a2 2 0 00-2-2H8a2 2 0 00-2 2v14a2 2 0 002 2z" />}
                    label="DEVICES" active={activeTab === "DEVICES"} 
                    onClick={() => setActiveTab("DEVICES")} 
                />
            </div>
        </div>

        {/* Presenter Bar — bottom-anchored so it never overlaps chat or mic */}
        <div className="fixed bottom-[88px] left-1/2 -translate-x-1/2 z-[80] flex items-center gap-1.5 px-2 py-2 rounded-full bg-white/80 backdrop-blur-md border border-slate-200 shadow-lg opacity-20 hover:opacity-100 transition-opacity duration-300 pointer-events-auto">
            <span className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-500 px-3 pl-4 border-r border-slate-300">Demo tools</span>
            <div className="flex items-center gap-1">
                {(['high', 'low', 'stable'] as const).map((s) => (
                    <button 
                        key={s}
                        onClick={() => setScenario(s)}
                        className={`text-[9px] font-black uppercase tracking-widest px-3 py-1.5 rounded-full transition-all ${scenario === s ? 'bg-zyntra-navy text-white' : 'text-slate-500 hover:bg-slate-200'}`}
                    >
                        {s}
                    </button>
                ))}
            </div>
            <div className="w-px h-4 bg-slate-300 mx-1" />
            <button 
                onClick={() => setShowDemoAlert(true)}
                className="text-[9px] font-black uppercase tracking-widest px-3 py-1.5 rounded-full text-slate-500 hover:bg-slate-200 hover:text-zyntra-navy"
            >
                Simulate Alert
            </button>
            <button 
                onClick={() => setVoiceAlertsEnabled(prev => !prev)}
                className={`text-[9px] font-black uppercase tracking-widest px-3 py-1.5 rounded-full transition-all ${voiceAlertsEnabled ? 'text-teal-600 font-black' : 'text-slate-400'}`}
            >
                Voice: {voiceAlertsEnabled ? 'ON' : 'OFF'}
            </button>
        </div>

        {/* Zyntra Chat Modal */}
        {isChatOpen && (
            <ZyntraChat initialOutput={zyntraData as any} onClose={() => setIsChatOpen(false)} />
        )}

        {/* Proactive Alert Simulation Layer */}
        {showDemoAlert && zyntraData && (
            <ZyntraAlertOverlay 
                payload={zyntraData.alertPayload} 
                onOpenApp={() => {
                    setShowDemoAlert(false);
                    // UX: Scroll to status card to reinforce connection
                    statusCardRef.current?.scrollIntoView({ behavior: 'smooth' });
                }} 
            />
        )}
        </>
    );
}

function NavIcon({ icon, label, active, onClick }: { icon: React.ReactNode, label: string, active: boolean, onClick: () => void }) {
    return (
        <button onClick={onClick} className={`flex flex-col items-center gap-1.5 transition-colors ${active ? "text-zyntra-navy" : "text-slate-400 hover:text-slate-600"}`}>
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                {icon}
            </svg>
            <span className="text-[9px] font-bold tracking-widest">{label}</span>
        </button>
    );
}
