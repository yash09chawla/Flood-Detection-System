"use client";

import type { HTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface CardProps extends HTMLAttributes<HTMLDivElement> {
  /** "solid" = opaque card, "glass" = backdrop-blurred panel */
  variant?: "solid" | "glass";
}

export function Card({ className, variant = "solid", ...props }: CardProps) {
  return (
    <div
      className={cn(
        "rounded-xl p-4",
        variant === "glass"
          ? "glass relative glass-sheen"
          : "bg-canvas/60 border border-border",
        className,
      )}
      {...props}
    />
  );
}

export function CardHeader({ className, ...props }: HTMLAttributes<HTMLDivElement>) {
  return (
    <div
      className={cn("mb-3 flex items-center justify-between gap-2", className)}
      {...props}
    />
  );
}

export function CardTitle({ className, ...props }: HTMLAttributes<HTMLHeadingElement>) {
  return (
    <h3
      className={cn(
        "text-[10px] font-semibold tracking-label uppercase text-muted",
        className,
      )}
      {...props}
    />
  );
}
