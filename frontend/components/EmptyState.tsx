"use client";

import { cn } from "@/lib/utils";

interface Props {
  title: string;
  description?: string;
  className?: string;
}

export function EmptyState({ title, description, className }: Props) {
  return (
    <div
      className={cn(
        "flex flex-col items-center justify-center text-center gap-4 py-10 px-6",
        className,
      )}
    >
      {/* Animated concentric ring as the icon */}
      <div className="relative w-14 h-14 grid place-items-center">
        <span
          className="absolute inset-0 rounded-full border border-line/15
                     animate-spin-slow"
          style={{
            background:
              "conic-gradient(from 0deg, rgb(var(--c-accent)) 0%, transparent 50%, transparent 100%)",
            mask: "radial-gradient(circle, transparent 60%, black 61%)",
            WebkitMask: "radial-gradient(circle, transparent 60%, black 61%)",
          }}
          aria-hidden
        />
        <span
          className="w-9 h-9 rounded-full bg-accent-soft border border-line/15"
          aria-hidden
        />
      </div>

      <div className="space-y-1">
        <p className="font-serif text-[15px] text-ink leading-tight">{title}</p>
        {description && (
          <p className="text-xs text-muted leading-relaxed max-w-[240px] mx-auto">
            {description}
          </p>
        )}
      </div>
    </div>
  );
}
