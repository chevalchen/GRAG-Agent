// ReactMarkdown component configuration (safe renderer for TCM answers)
import remarkGfm from "remark-gfm";

export const MARKDOWN_PLUGINS = [remarkGfm];

// Allowed HTML elements — we rely on react-markdown's safe defaults
// (no raw HTML pass-through, so XSS is not a concern here)
export const MARKDOWN_CLASS = "markdown-body text-paper text-sm leading-relaxed";
