"use client";

import { cn } from "@/lib/utils";

interface BrandMarkProps {
  className?: string;
  showWordmark?: boolean;
  size?: "sm" | "md";
}

export function BrandMark({
  className,
  showWordmark = true,
  size = "md",
}: BrandMarkProps) {
  const iconSize = size === "sm" ? 22 : 28;
  return (
    <div className={cn("inline-flex items-center gap-2.5", className)}>
      <span
        className="relative inline-flex items-center justify-center rounded-lg
                   bg-accent-grad text-white shadow-soft"
        style={{ width: iconSize, height: iconSize }}
      >
        {/* Stylised water-drop monogram */}
        <svg
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth={2}
          strokeLinecap="round"
          strokeLinejoin="round"
          className="w-[60%] h-[60%]"
          aria-hidden
        >
          <path d="M12 3.2c2.8 3.6 5.2 6.7 5.2 9.6a5.2 5.2 0 1 1-10.4 0c0-2.9 2.4-6 5.2-9.6Z" />
        </svg>
      </span>
      {showWordmark && (
        <span
          className={cn(
            "font-serif tracking-tight text-ink",
            size === "sm" ? "text-[15px] leading-none" : "text-lg leading-none",
          )}
        >
          FloodMap
        </span>
      )}
    </div>
  );
}
