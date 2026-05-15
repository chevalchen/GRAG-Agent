// Root layout: collapsible Sidebar + ChatArea
import { useState } from "react";
import { Sidebar } from "./components/Sidebar";
import { ChatArea } from "./components/ChatArea";

export default function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);

  return (
    <div className="flex h-full bg-base">
      <Sidebar isOpen={sidebarOpen} onClose={() => setSidebarOpen(false)} />

      {/* main content */}
      <div className="flex flex-col flex-1 min-w-0 h-full">
        {/* mobile menu toggle */}
        <button
          onClick={() => setSidebarOpen(true)}
          className="
            md:hidden absolute top-3 left-3 z-30
            w-8 h-8 flex items-center justify-center
            text-muted hover:text-accent transition-colors
          "
          aria-label="打开侧边栏"
        >
          ☰
        </button>

        <ChatArea />
      </div>
    </div>
  );
}
