import { useState } from "react";
import type { ServerVersionMeta } from "../lib/types";

type DebugDrawerProps = {
  data: unknown;
  rawStream: string;
  serverVersion: ServerVersionMeta | null;
};

export default function DebugDrawer({ data, rawStream, serverVersion }: DebugDrawerProps) {
  const [tab, setTab] = useState<"json" | "stream">("json");
  const isDebugMode = import.meta.env.VITE_DEBUG_MODE === "true";

  if (!data && !rawStream && !serverVersion) return null;

  return (
    <div className="bg-black/40 rounded-lg overflow-hidden border border-white/10 mt-4">
      <div className="flex bg-black/50 border-b border-white/10 text-xs">
        <button 
          onClick={() => setTab("json")}
          className={`flex-1 py-2 ${tab === "json" ? "bg-white/10 font-bold" : "text-slate-400 hover:bg-white/5"}`}
        >
          Debug JSON
        </button>
        {isDebugMode && (
          <button 
            onClick={() => setTab("stream")}
            className={`flex-1 py-2 ${tab === "stream" ? "bg-white/10 font-bold" : "text-slate-400 hover:bg-white/5"}`}
          >
            Raw Stream
          </button>
        )}
      </div>

      <div className="p-3 text-xs overflow-auto max-h-64 font-mono text-slate-300">
        {tab === "json" ? (
          <pre>{JSON.stringify(data, null, 2)}</pre>
        ) : (
          <pre className="whitespace-pre-wrap">{rawStream || "No stream data yet..."}</pre>
        )}
      </div>
      
      {serverVersion && (
        <div className="bg-black/60 p-2 text-[10px] text-slate-500 text-right border-t border-white/5">
          Commit: {serverVersion.git_commit || "unknown"} | Build Time: {serverVersion.file_mtime ? new Date(serverVersion.file_mtime * 1000).toLocaleString() : "unknown"}
        </div>
      )}
    </div>
  );
}
