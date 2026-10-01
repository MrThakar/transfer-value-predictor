// Typed client for the FastAPI backend (see transfer_value/api.py).

export type Position = "Goalkeeper" | "Defender" | "Midfield" | "Attack";

export interface PlayerSeason {
  age: number;
  position: Position;
  sub_position: string;
  league: string;
  appearances: number;
  minutes_played: number;
  goals: number;
  assists: number;
  yellow_cards: number;
  prev_minutes: number;
  prev_goals: number;
  prev_assists: number;
  club_points_per_game: number;
  club_goal_diff_per_game: number;
}

export interface Prediction {
  predicted_value_eur: number;
  model: string;
}

export interface Comparable {
  name: string;
  league: string;
  position: Position;
  age: number;
  goals: number;
  assists: number;
  market_value_in_eur: number;
}

export interface Metadata {
  model: string;
  seasons: number[];
  leagues: Record<string, string>;
  positions: Record<Position, string[]>;
  holdout_median_pct_error: number | null;
  holdout_season: number | null;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    const detail = typeof body?.detail === "string" ? body.detail : "Request failed";
    throw new Error(`${detail} (${response.status})`);
  }
  return response.json() as Promise<T>;
}

export const getMetadata = () => request<Metadata>("/metadata");

export const predict = (player: PlayerSeason, signal?: AbortSignal) =>
  request<Prediction>("/predict", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(player),
    signal,
  });

export const getComparables = (value: number, position: Position, signal?: AbortSignal) =>
  request<Comparable[]>(
    `/comparables?value=${Math.round(value)}&position=${encodeURIComponent(position)}&limit=5`,
    { signal },
  );
