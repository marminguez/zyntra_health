import React from "react";

interface ActionCardProps {
    title: string;
    description: string;
    icon?: React.ReactNode;
    onClick?: () => void;
}

export function ActionCard({ title, description, icon, onClick }: ActionCardProps) {
    return (
        <div 
            onClick={onClick}
            className={`bg-white rounded-[1.25rem] p-5 shadow-sm border border-slate-100 flex items-start gap-4 transition-all duration-200 ${onClick ? 'cursor-pointer hover:shadow-md hover:-translate-y-0.5' : ''}`}
        >
            {icon && (
                <div className="flex-shrink-0 w-10 h-10 bg-slate-50 rounded-full flex items-center justify-center text-slate-400">
                    {icon}
                </div>
            )}
            <div className="flex-1">
                <h4 className="font-bold text-slate-900 mb-1">{title}</h4>
                <p className="text-slate-500 text-sm leading-relaxed">{description}</p>
            </div>
            {onClick && (
                <div className="flex-shrink-0 pt-1 text-slate-300">
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                    </svg>
                </div>
            )}
        </div>
    );
}
