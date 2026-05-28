"use client";

import { useEffect, useState } from "react";
import { Moon, Sun } from "lucide-react";
import { cn } from "@/lib/utils";

type Theme = "light" | "dark";

export function ThemeToggle({ className }: { className?: string }) {
  const [theme, setTheme] = useState<Theme>("light");
  const [mounted, setMounted] = useState(false);

  // Read the theme set by the inline pre-paint script in layout.tsx
  useEffect(() => {
    const isDark = document.documentElement.classList.contains("dark");
    setTheme(isDark ? "dark" : "light");
    setMounted(true);
  }, []);

  const toggle = () => {
    const next = theme === "light" ? "dark" : "light";
    document.documentElement.classList.toggle("dark", next === "dark");
    try { localStorage.setItem("theme", next); } catch {}
    setTheme(next);
  };

  // Avoid hydration mismatch — render an empty placeholder until client mount
  if (!mounted) {
    return (
      <button
        aria-hidden
        className={cn(
          "glass-pill h-10 w-10 rounded-full grid place-items-center",
          className,
        )}
      />
    );
  }

  return (
    <button
      type="button"
      onClick={toggle}
      aria-label={`Switch to ${theme === "light" ? "dark" : "light"} theme`}
      className={cn(
        "glass-pill h-10 w-10 rounded-full grid place-items-center",
        "text-text hover:text-ink transition-colors duration-200 cursor-pointer",
        "focus-visible:outline-none focus-visible:shadow-focus",
        className,
      )}
    >
      {theme === "light" ? (
        <Moon size={16} strokeWidth={1.75} />
      ) : (
        <Sun size={16} strokeWidth={1.75} />
      )}
    </button>
  );
}
