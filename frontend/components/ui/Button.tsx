"use client";

import { forwardRef, type ButtonHTMLAttributes } from "react";
import { cn } from "@/lib/utils";

type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  fullWidth?: boolean;
}

const sizeStyles: Record<Size, string> = {
  sm: "h-8  px-3   text-xs   rounded-md gap-1.5",
  md: "h-10 px-4   text-sm   rounded-lg gap-2",
  lg: "h-11 px-5   text-sm   rounded-lg gap-2",
};

const variantStyles: Record<Variant, string> = {
  primary:
    "bg-accent-grad text-white shadow-cta " +
    "hover:brightness-110 active:scale-[0.985] " +
    "focus-visible:shadow-focus",
  secondary:
    "bg-canvas border border-border text-ink " +
    "hover:bg-accent-soft hover:border-accent/30 " +
    "focus-visible:shadow-focus",
  ghost:
    "bg-transparent text-text " +
    "hover:bg-line/5 hover:text-ink " +
    "focus-visible:shadow-focus",
  danger:
    "bg-flood text-white " +
    "hover:brightness-110 active:scale-[0.985] " +
    "focus-visible:shadow-focus",
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = "primary", size = "md", fullWidth, ...props }, ref) => (
    <button
      ref={ref}
      className={cn(
        "inline-flex items-center justify-center font-medium",
        "transition-all duration-200 ease-out-expo outline-none cursor-pointer",
        "disabled:opacity-50 disabled:cursor-not-allowed disabled:active:scale-100",
        sizeStyles[size],
        variantStyles[variant],
        fullWidth && "w-full",
        className,
      )}
      {...props}
    />
  ),
);
Button.displayName = "Button";
