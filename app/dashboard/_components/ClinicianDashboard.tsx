"use client";

import { useSession, signOut } from "next-auth/react";
import { useState, useEffect } from "react";
import { useRouter, useSearchParams } from "next/navigation";

export interface Driver {
    feature: string;
    contribution: number;
    direction: "up" | "down";
}

export interface RiskScoreItem {
    id?: string;
    horizonHrs: number;
    score: number;
    level: "low" | "medium" | "high";
    explanation: string;
    confidence?: number;
    featuresUsed?: string[];
    modelVersion?: string;
}

export interface MLPayload {
    ok: boolean;
    usedModel: "ml" | "zscore";
    pci?: number;
    global_24?: number;
    global_72?: number;
    p_hyper_24?: number;
    p_hyper_72?: number;
    p_hypo_12?: number;
    p_hypo_24?: number;
    p_instability_24?: number;
    p_instability_72?: number;
    needs_human_review?: boolean;
    warning?: string;
    drivers?: {
        hyper_24: Driver[];
        hypo_12: Driver[];
        instability_24: Driver[];
    };
    riskScores?: RiskScoreItem[];
}

export function ClinicianDashboard() {
    const { data: session } = useSession();
    const router = useRouter();
    const searchParams = useSearchParams();
    const [patientId, setPatientId] = useState("");
    const [payload, setPayload] = useState<MLPayload | null>(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState("");

    const [isSyncingFitbit, setIsSyncingFitbit] = useState(false);
    const [isSyncingLibre, setIsSyncingLibre] = useState(false);
    const [syncMessage, setSyncMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

    useEffect(() => {
        if (searchParams?.get("fitbit") === "connected") {
            setSyncMessage({ type: "success", text: "Fitbit account connected successfully!" });
            router.replace("/dashboard", { scroll: false });
        }
    }, [searchParams, router]);

    async function fetchRisk() {
        if (!patientId.trim()) return;
        setLoading(true);
        setError("");
        setPayload(null);
        setSyncMessage(null);

        try {
            const res = await fetch("/app-api/risk", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ patientId }),
            });

            const data = await res.json();
            if (!res.ok) {
                setError(data.error || "Request failed");
            } else {
                setPayload(data);
            }
        } catch {
            setError("Network error");
        } finally {
            setLoading(false);
        }
    }

    async function handleSyncFitbit() {
        if (!patientId.trim()) return;
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
                setSyncMessage({ type: "success", text: "Fitbit synced successfully" });
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
        if (!patientId.trim()) return;
        setIsSyncingLibre(true);
        setSyncMessage(null);
        try {
            const res = await fetch("/app-api/integrations/freestyle/sync", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                credentials: "include",
                body: JSON.stringify({ patientId }),
            });
            const data = await res.json();
            if (res.ok) {
                setSyncMessage({ type: "success", text: "FreeStyle CGM synced successfully" });
            } else {
                setSyncMessage({ type: "error", text: `Sync failed: ${data.error || "Unknown error"}` });
            }
        } catch (err) {
            setSyncMessage({ type: "error", text: "Sync failed: Network error" });
        } finally {
            setIsSyncingLibre(false);
        }
    }

    const renderDrivers = (drivers: Driver[]) => {
        if (!drivers || drivers.length === 0) return <p className="text-slate-500 text-sm">No significant drivers</p>;
        return (
            <ul className="text-sm text-slate-500 mt-2 space-y-1">
                {drivers.map((d, i) => (
                    <li key={i} className="flex items-center gap-2">
                        <span className={`font-bold ${d.direction === "up" ? "text-red-500" : "text-green-500"}`}>
                            {d.direction === "up" ? "↑" : "↓"}
                        </span>
                        {d.feature.replace("delta_", "Δ ")}
                    </li>
                ))}
            </ul>
        );
    };

    return (
        <div className="min-h-screen bg-slate-50 text-slate-900 font-sans">
            <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 pt-8 pb-12">
                <header className="flex items-center justify-between mb-8">
                    <div>
                        <h1 className="text-2xl font-bold text-slate-900">Zyntra Intelligence</h1>
                        <p className="text-slate-500 m-0">
                            Diabetes Predictive Layer · {(session?.user as any)?.role}
                        </p>
                    </div>
                    <button 
                        aria-label="Sign out of Zyntra"
                        className="bg-teal-600 hover:bg-teal-700 text-white font-medium rounded-lg px-4 py-2 transition-colors min-h-[44px] min-w-[44px] focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2"
                        onClick={() => signOut({ callbackUrl: "/login" })}
                    >
                        Sign Out
                    </button>
                </header>

                <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
                    <h2 className="text-slate-800 font-semibold text-lg mb-4">Patient</h2>
                    <div className="flex gap-3 mb-6">
                        <label htmlFor="patient-id" className="sr-only">Patient ID</label>
                        <input
                            id="patient-id"
                            className="border border-slate-300 rounded-lg px-3 py-2 text-slate-900 bg-white focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2 w-full max-w-sm min-h-[44px]"
                            placeholder="Patient ID"
                            value={patientId}
                            onChange={(e) => setPatientId(e.target.value)}
                        />
                        <button 
                            aria-label="Calculate Risk for this patient"
                            className="bg-teal-600 hover:bg-teal-700 text-white font-medium rounded-lg px-4 py-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px] min-w-[44px] focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2 flex items-center justify-center"
                            onClick={fetchRisk}
                            disabled={loading || !patientId.trim()}
                            aria-busy={loading}
                        >
                            {loading ? (
                                <>
                                    <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                                        <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                        <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                    </svg>
                                    Calculating risk...
                                </>
                            ) : "Calculate Risk"}
                        </button>
                    </div>

                    {error && <p className="text-red-700 bg-red-50 border border-red-200 rounded-lg px-3 py-2 mb-6">{error}</p>}

                    <div className="border-t border-slate-100 pt-6">
                        <h2 className="text-slate-800 font-semibold text-lg mb-4">Data Sources & Integrations</h2>
                        
                        <div className="flex flex-wrap gap-3 items-center">
                            <a 
                                href="/app-api/integrations/fitbit/connect"
                                aria-label="Connect Fitbit account"
                                className="border border-teal-600 text-teal-600 hover:bg-teal-50 font-medium rounded-lg px-4 py-2 transition-colors min-h-[44px] min-w-[44px] inline-flex items-center justify-center focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2"
                            >
                                Connect Fitbit
                            </a>

                            <button
                                aria-label="Sync Fitbit data for this patient"
                                className="bg-teal-600 hover:bg-teal-700 text-white font-medium rounded-lg px-4 py-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px] min-w-[44px] focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2 flex items-center justify-center"
                                onClick={handleSyncFitbit}
                                disabled={isSyncingFitbit || !patientId.trim()}
                                aria-busy={isSyncingFitbit}
                            >
                                {isSyncingFitbit ? (
                                    <>
                                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Syncing...
                                    </>
                                ) : "Sync Fitbit"}
                            </button>

                            <button
                                aria-label="Sync FreeStyle CGM data for this patient"
                                className="bg-teal-600 hover:bg-teal-700 text-white font-medium rounded-lg px-4 py-2 transition-colors disabled:opacity-50 disabled:cursor-not-allowed min-h-[44px] min-w-[44px] focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2 flex items-center justify-center"
                                onClick={handleSyncLibre}
                                disabled={isSyncingLibre || !patientId.trim()}
                                aria-busy={isSyncingLibre}
                            >
                                {isSyncingLibre ? (
                                    <>
                                        <svg className="animate-spin -ml-1 mr-2 h-4 w-4 text-white" fill="none" viewBox="0 0 24 24">
                                            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                                            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                                        </svg>
                                        Syncing...
                                    </>
                                ) : "Sync FreeStyle CGM"}
                            </button>
                        </div>
                        
                        {syncMessage && (
                            <p className={`mt-4 ${syncMessage.type === "success" ? "text-green-700 bg-green-50 border border-green-200" : "text-red-700 bg-red-50 border border-red-200"} rounded-lg px-3 py-2 max-w-lg`}>
                                {syncMessage.text}
                            </p>
                        )}
                    </div>
                </div>

                {payload && (
                    <div className="bg-amber-50 border border-amber-200 text-amber-800 rounded-lg p-4 mb-6">
                        <strong>Disclaimer:</strong> Zyntra does not make clinical decisions. Human supervision required.
                    </div>
                )}

                {payload && payload.usedModel === "ml" && (
                    <>
                        {/* PCI Visualizer */}
                        <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-6 mb-6">
                            <h3 className="text-slate-800 font-semibold text-lg mb-3">Physiological Coherence Index (PCI)</h3>
                            <div className="flex items-center gap-4">
                                <div className="flex-1 h-3 rounded-full overflow-hidden relative" style={{ background: "linear-gradient(90deg, #52c41a, #faad14, #ff4d4f, #cf1322)" }}>
                                    <div className="absolute top-0 bottom-0 w-1 bg-white border border-black" style={{ left: `${(payload.pci ?? 0) * 100}%` }} />
                                </div>
                                <span className="font-bold text-xl text-slate-900 w-16 text-right">
                                    {((payload.pci ?? 0) * 100).toFixed(1)}
                                </span>
                            </div>
                            <p className="text-sm text-slate-500 mt-2">0 = High Coherence, 100 = Systemic Strain</p>
                        </div>

                        {/* 4 Cards Grid */}
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 xl:grid-cols-4 gap-6">
                            {/* 1. Hyper Risk */}
                            <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
                                <h3 className="text-slate-800 font-semibold text-lg mb-4">Hyper Risk</h3>
                                <div className="flex justify-between mb-4">
                                    <div>
                                        <p className="text-sm text-slate-500 mb-1">24h Horizon</p>
                                        <p className="text-xl font-bold text-slate-900">{((payload.p_hyper_24 ?? 0) * 100).toFixed(1)}%</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-slate-500 mb-1">72h Horizon</p>
                                        <p className="text-xl font-bold text-slate-900">{((payload.p_hyper_72 ?? 0) * 100).toFixed(1)}%</p>
                                    </div>
                                </div>
                                <p className="font-semibold text-sm text-slate-900">Why this?</p>
                                {renderDrivers(payload.drivers?.hyper_24 ?? [])}
                            </div>

                            {/* 2. Hypo Risk */}
                            <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
                                <h3 className="text-slate-800 font-semibold text-lg mb-4">Hypo Risk</h3>
                                <div className="flex justify-between mb-4">
                                    <div>
                                        <p className="text-sm text-slate-500 mb-1">12h Horizon</p>
                                        <p className="text-xl font-bold text-slate-900">{((payload.p_hypo_12 ?? 0) * 100).toFixed(1)}%</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-slate-500 mb-1">24h Horizon</p>
                                        <p className="text-xl font-bold text-slate-900">{((payload.p_hypo_24 ?? 0) * 100).toFixed(1)}%</p>
                                    </div>
                                </div>
                                <p className="font-semibold text-sm text-slate-900">Why this?</p>
                                {renderDrivers(payload.drivers?.hypo_12 ?? [])}
                            </div>

                            {/* 3. Global Instability */}
                            <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-5">
                                <h3 className="text-slate-800 font-semibold text-lg mb-4">Global Instability</h3>
                                <div className="flex justify-between mb-4">
                                    <div>
                                        <p className="text-sm text-slate-500 mb-1">24h Horizon</p>
                                        <p className="text-xl font-bold text-slate-900">{((payload.global_24 ?? 0) * 100).toFixed(1)}%</p>
                                    </div>
                                    <div>
                                        <p className="text-sm text-slate-500 mb-1">72h Horizon</p>
                                        <p className="text-xl font-bold text-slate-900">{((payload.global_72 ?? 0) * 100).toFixed(1)}%</p>
                                    </div>
                                </div>
                                <p className="font-semibold text-sm text-slate-900">Why this?</p>
                                {renderDrivers(payload.drivers?.instability_24 ?? [])}
                            </div>

                            {/* 4. Needs Human Review */}
                            <div className={`border rounded-xl shadow-sm p-5 ${payload.needs_human_review ? 'bg-red-50 border-red-200' : 'bg-white border-slate-200'}`}>
                                <h3 className={`font-semibold text-lg mb-4 ${payload.needs_human_review ? 'text-red-700' : 'text-slate-800'}`}>Needs Human Review</h3>
                                <div className="mb-4 flex items-center gap-2">
                                    <span className={`px-3 py-1.5 rounded-full text-sm font-semibold ${payload.needs_human_review ? 'bg-red-100 text-red-800' : 'bg-green-100 text-green-800'}`}>
                                        {payload.needs_human_review ? "FLAGGED" : "CLEAR"}
                                    </span>
                                </div>
                                {payload.needs_human_review && payload.warning && (
                                    <p className="text-red-700 text-sm font-semibold mt-2">
                                        {payload.warning}
                                    </p>
                                )}
                            </div>
                        </div>
                    </>
                )}

                {payload && payload.usedModel === "zscore" && (
                    <div>
                        <h2 className="text-slate-800 font-semibold text-lg mb-4">Risk Assessment</h2>
                        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                            {payload.riskScores?.map((rs, idx) => {
                                const confPct = Math.round((rs.confidence ?? 1.0) * 100);
                                let confColorClass = "bg-red-100 text-red-800";
                                if (confPct >= 80) confColorClass = "bg-green-100 text-green-800";
                                else if (confPct >= 50) confColorClass = "bg-amber-100 text-amber-800";

                                let levelColorClass = "bg-green-100 text-green-800";
                                if (rs.level === "high") levelColorClass = "bg-red-100 text-red-800";
                                else if (rs.level === "medium") levelColorClass = "bg-amber-100 text-amber-800";

                                const levelCapitalized = rs.level.charAt(0).toUpperCase() + rs.level.slice(1);

                                return (
                                    <div key={idx} role="region" aria-label={`Risk score for ${rs.horizonHrs}h horizon`} className="bg-white border border-slate-200 rounded-xl shadow-sm p-5 flex flex-col">
                                        <h3 className="text-slate-500 font-medium text-xs uppercase tracking-wider mb-2">Horizon</h3>
                                        <p className="text-2xl font-bold text-slate-900 mb-4">{rs.horizonHrs}h</p>

                                        <div className="flex items-center justify-between mb-3">
                                            <span className="text-slate-500 text-sm">Risk Level</span>
                                            <span className={`px-2.5 py-1 rounded-full text-xs font-semibold ${levelColorClass}`}>
                                                {levelCapitalized}
                                            </span>
                                        </div>

                                        <div className="flex items-center justify-between mb-4">
                                            <span className="text-slate-500 text-sm">Score</span>
                                            <span className="text-slate-900 font-bold">{(rs.score * 100).toFixed(1)}%</span>
                                        </div>

                                        <div className="flex items-center justify-between mb-4 pb-4 border-b border-slate-100">
                                            <span className="text-slate-500 text-sm">Model Confidence</span>
                                            <span aria-label={`Model confidence: ${confPct}%`} className={`px-2.5 py-1 rounded-full text-xs font-semibold ${confColorClass}`}>
                                                {confPct}%
                                            </span>
                                        </div>

                                        <div className="mb-4 flex-grow">
                                            <span className="text-slate-500 text-sm block mb-1">Explanation</span>
                                            <p className="text-slate-700 text-sm leading-relaxed">{rs.explanation}</p>
                                        </div>

                                        {rs.featuresUsed && rs.featuresUsed.length > 0 && (
                                            <details className="mb-4 group">
                                                <summary className="text-slate-700 text-sm font-medium cursor-pointer hover:text-slate-900 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 rounded ring-offset-2">
                                                    Features used by the model
                                                </summary>
                                                <ul className="mt-3 text-sm text-slate-600 list-disc pl-5 space-y-1 bg-slate-50 p-3 rounded-lg border border-slate-100">
                                                    {rs.featuresUsed.map(f => (
                                                        <li key={f}>{f}</li>
                                                    ))}
                                                </ul>
                                            </details>
                                        )}

                                        {rs.modelVersion && (
                                            <div className="mt-auto pt-4 border-t border-slate-100">
                                                <span className="text-slate-400 text-xs">Model version: {rs.modelVersion}</span>
                                            </div>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    </div>
                )}

                {!payload && !error && !loading && (
                    <div className="bg-white border border-slate-200 rounded-xl shadow-sm p-8 text-center">
                        <p className="text-slate-500 text-lg">
                            Enter a patient ID and click <strong className="text-slate-700 font-semibold">Calculate Risk</strong> to see results.
                        </p>
                    </div>
                )}
            </div>
        </div>
    );
}
