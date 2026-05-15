// Base fetch wrapper — handles the {ok, data, error, trace_id} envelope
const BASE = "/api/v1";

export class ApiError extends Error {
  code: string;
  traceId?: string;

  constructor(code: string, message: string, traceId?: string) {
    super(message);
    this.name = "ApiError";
    this.code = code;
    this.traceId = traceId;
  }
}

export async function apiFetch<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json", ...options.headers },
    ...options,
  });

  const traceId = res.headers.get("x-trace-id") ?? undefined;

  if (!res.ok) {
    let code = "HTTP_ERROR";
    let message = `HTTP ${res.status}`;
    try {
      const body = await res.json();
      code = body?.error?.code ?? code;
      message = body?.error?.message ?? message;
    } catch {
      // ignore parse errors
    }
    throw new ApiError(code, message, traceId);
  }

  const body = await res.json();
  if (!body.ok) {
    throw new ApiError(
      body.error?.code ?? "INTERNAL_ERROR",
      body.error?.message ?? "Unknown error",
      traceId,
    );
  }
  return body.data as T;
}
