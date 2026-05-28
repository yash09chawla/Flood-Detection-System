"use client";

import { forwardRef, type InputHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  invalid?: boolean;
  /** When true, render with monospace digits (for coordinates etc) */
  mono?: boolean;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ className, label, hint, invalid, mono, id, ...props }, ref) => {
    return (
      <label className="flex flex-col gap-1.5 group">
        {label && (
          <span className="text-[10px] font-medium tracking-label uppercase text-muted">
            {label}
          </span>
        )}
        <input
          ref={ref}
          id={id}
          className={cn(
            "h-10 px-3 rounded-lg text-sm",
            "bg-canvas border border-border text-ink",
            "outline-none transition-all duration-200",
            "placeholder:text-subtle",
            "hover:border-accent/30",
            "focus-visible:border-accent focus-visible:shadow-focus",
            mono && "font-mono tabular text-[13px]",
            invalid &&
              "border-danger/60 focus-visible:border-danger focus-visible:shadow-[0_0_0_3px_rgb(var(--c-danger)/0.22)]",
            className,
          )}
          {...props}
        />
        {hint && (
          <span className="text-[11px] text-subtle leading-snug">{hint}</span>
        )}
      </label>
    );
  },
);
Input.displayName = "Input";
