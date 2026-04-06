import React from "react";
import { RiskLevel } from "../../_lib/mockRiskData";

interface RiskBadgeProps {
    level: RiskLevel;
}

export function RiskBadge({ level }: RiskBadgeProps) {
    if (level === "low") {
        return (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold leading-5 bg-teal-50 text-teal-700 border border-teal-100 uppercase tracking-wider">
                Low
            </span>
        );
    }
    
    if (level === "medium") {
        return (
            <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold leading-5 bg-amber-50 text-amber-700 border border-amber-100 uppercase tracking-wider">
                Medium
            </span>
        );
    }
    
    return (
        <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold leading-5 bg-rose-50 text-rose-700 border border-rose-100 uppercase tracking-wider">
            High
        </span>
    );
}
