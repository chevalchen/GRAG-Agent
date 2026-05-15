// Session management API calls
import { apiFetch } from "./client";

export interface SessionEntry {
  session_id: string;
  checkpoint_id?: string;
}

export interface SessionsResponse {
  sessions: SessionEntry[];
}

export async function fetchSessions(): Promise<SessionEntry[]> {
  const data = await apiFetch<SessionsResponse>("/sessions");
  return data.sessions ?? [];
}

export async function deleteSession(sessionId: string): Promise<void> {
  await apiFetch<unknown>(`/sessions/${sessionId}`, { method: "DELETE" });
}
