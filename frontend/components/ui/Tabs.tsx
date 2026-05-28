"use client";

import { useEffect, useRef, useState } from "react";
import { cn } from "@/lib/utils";

interface TabsProps<T extends string> {
  value: T;
  onChange: (value: T) => void;
  options: { value: T; label: string; icon?: React.ReactNode }[];
  className?: string;
}

/**
 * Pill-style segmented control with a smooth sliding indicator that animates
 * between the active tab positions. Uses a single absolutely-positioned thumb
 * that translates + resizes — much smoother than re-painting the active state
 * on different siblings.
 */
export function Tabs<T extends string>({
  value,
  onChange,
  options,
  className,
}: TabsProps<T>) {
  const containerRef = useRef<HTMLDivElement>(null);
  const buttonRefs = useRef<Record<string, HTMLButtonElement | null>>({});
  const [thumb, setThumb] = useState<{ x: number; w: number } | null>(null);

  // Recompute thumb position whenever the value changes or the container resizes
  useEffect(() => {
    const container = containerRef.current;
    const btn = buttonRefs.current[value];
    if (!container || !btn) return;
    const cRect = container.getBoundingClientRect();
    const bRect = btn.getBoundingClientRect();
    setThumb({ x: bRect.left - cRect.left, w: bRect.width });
  }, [value, options.length]);

  return (
    <div
      ref={containerRef}
      role="tablist"
      className={cn(
        "relative inline-flex p-1 rounded-lg",
        "bg-line/[0.06] border border-border",
        className,
      )}
    >
      {/* Sliding thumb */}
      {thumb && (
        <span
          aria-hidden
          className="absolute top-1 bottom-1 rounded-md bg-canvas shadow-soft
                     border border-border transition-all duration-300 ease-out-expo"
          style={{ transform: `translateX(${thumb.x - 4}px)`, width: thumb.w }}
        />
      )}

      {options.map((opt) => {
        const active = opt.value === value;
        return (
          <button
            key={opt.value}
            ref={(el) => { buttonRefs.current[opt.value] = el; }}
            role="tab"
            type="button"
            aria-selected={active}
            onClick={() => onChange(opt.value)}
            className={cn(
              "relative z-10 px-3 h-8 inline-flex items-center gap-1.5 rounded-md",
              "text-xs font-medium transition-colors duration-200 cursor-pointer",
              "outline-none focus-visible:shadow-focus",
              active ? "text-ink" : "text-muted hover:text-text",
            )}
          >
            {opt.icon}
            {opt.label}
          </button>
        );
      })}
    </div>
  );
}
