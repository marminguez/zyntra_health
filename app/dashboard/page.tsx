"use client";

import { useSession } from "next-auth/react";
import { useRouter } from "next/navigation";
import { ClinicianDashboard } from "./_components/ClinicianDashboard";
import { PatientDashboard } from "./_components/PatientDashboard";

export default function DashboardPage() {
    const { data: session, status } = useSession();
    const router = useRouter();

    if (status === "loading") {
        return (
            <div className="min-h-screen bg-slate-50 flex items-center justify-center">
                <p className="text-slate-500 font-medium font-sans">Loading Zyntra...</p>
            </div>
        );
    }

    if (!session) {
        router.push("/login");
        return null;
    }

    const role = (session?.user as any)?.role;

    if (role === "PATIENT") {
        return <PatientDashboard />;
    }

    return <ClinicianDashboard />;
}
