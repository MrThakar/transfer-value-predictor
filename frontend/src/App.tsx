import { useEffect, useState } from "react";
import {
  getComparables,
  getMetadata,
  predict,
  type Comparable,
  type Metadata,
  type PlayerSeason,
  type Position,
  type Prediction,
} from "./api";

// Users think in "title contender" terms, not points per game, so the form
// offers tiers and maps them to the two club-strength features.
const CLUB_TIERS = [
  { label: "Title contender", points: 2.3, goalDiff: 1.3 },
  { label: "European places", points: 1.8, goalDiff: 0.5 },
  { label: "Mid-table", points: 1.35, goalDiff: 0 },
  { label: "Relegation battle", points: 0.9, goalDiff: -0.7 },
];

interface FormState extends Omit<PlayerSeason, "club_points_per_game" | "club_goal_diff_per_game"> {
  clubTier: number;
}

const PRESETS: { label: string; form: FormState }[] = [
  {
    label: "Elite striker",
    form: {
      age: 25, position: "Attack", sub_position: "Centre-Forward", league: "GB1",
      appearances: 35, minutes_played: 2950, goals: 27, assists: 8, yellow_cards: 3,
      prev_minutes: 2700, prev_goals: 22, prev_assists: 5, clubTier: 0,
    },
  },
  {
    label: "Teenage winger",
    form: {
      age: 19, position: "Attack", sub_position: "Right Winger", league: "ES1",
      appearances: 30, minutes_played: 2100, goals: 9, assists: 10, yellow_cards: 2,
      prev_minutes: 600, prev_goals: 2, prev_assists: 1, clubTier: 1,
    },
  },
  {
    label: "Midfield anchor",
    form: {
      age: 27, position: "Midfield", sub_position: "Defensive Midfield", league: "IT1",
      appearances: 33, minutes_played: 2800, goals: 2, assists: 4, yellow_cards: 8,
      prev_minutes: 2600, prev_goals: 1, prev_assists: 3, clubTier: 2,
    },
  },
  {
    label: "Veteran centre-back",
    form: {
      age: 34, position: "Defender", sub_position: "Centre-Back", league: "L1",
      appearances: 24, minutes_played: 2000, goals: 1, assists: 0, yellow_cards: 5,
      prev_minutes: 2400, prev_goals: 2, prev_assists: 1, clubTier: 3,
    },
  },
];

const PLURALS: Record<Position, string> = {
  Goalkeeper: "Goalkeepers",
  Defender: "Defenders",
  Midfield: "Midfielders",
  Attack: "Attackers",
};

const euros = (value: number) =>
  value >= 1e6
    ? `€${(value / 1e6).toFixed(value >= 1e7 ? 0 : 1)}m`
    : `€${Math.round(value / 1e3)}k`;

const inputClass =
  "w-full rounded-lg border border-slate-300 bg-white px-3 py-2 text-sm tabular-nums " +
  "focus:border-emerald-600 focus:outline-none focus:ring-2 focus:ring-emerald-600/30 " +
  "dark:border-slate-700 dark:bg-slate-900";

function NumberField(props: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (value: number) => void;
}) {
  const { label, value, min, max, onChange } = props;
  return (
    <label className="block">
      <span className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-400">
        {label}
      </span>
      <input
        type="number"
        inputMode="numeric"
        className={inputClass}
        value={value}
        min={min}
        max={max}
        onChange={(event) => {
          const parsed = Math.round(Number(event.target.value));
          onChange(Math.min(max, Math.max(min, Number.isFinite(parsed) ? parsed : min)));
        }}
      />
    </label>
  );
}

function Section(props: { title: string; children: React.ReactNode }) {
  return (
    <fieldset className="space-y-3">
      <legend className="text-sm font-semibold">{props.title}</legend>
      {props.children}
    </fieldset>
  );
}

export default function App() {
  const [metadata, setMetadata] = useState<Metadata | null>(null);
  const [form, setForm] = useState<FormState>(PRESETS[0].form);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [comparables, setComparables] = useState<Comparable[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const update = (changes: Partial<FormState>) => setForm((current) => ({ ...current, ...changes }));

  useEffect(() => {
    getMetadata()
      .then(setMetadata)
      .catch((reason: Error) => setError(reason.message));
  }, []);

  // Re-predict shortly after the form stops changing; abort stale requests.
  useEffect(() => {
    const controller = new AbortController();
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const { clubTier, ...stats } = form;
        const tier = CLUB_TIERS[clubTier];
        const result = await predict(
          { ...stats, club_points_per_game: tier.points, club_goal_diff_per_game: tier.goalDiff },
          controller.signal,
        );
        const similar = await getComparables(
          result.predicted_value_eur,
          form.position,
          controller.signal,
        );
        setPrediction(result);
        setComparables(similar);
        setError(null);
      } catch (reason) {
        if ((reason as Error).name !== "AbortError") setError((reason as Error).message);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }, 250);
    return () => {
      clearTimeout(timer);
      controller.abort();
    };
  }, [form]);

  const positions = metadata ? (Object.keys(metadata.positions) as Position[]) : [form.position];
  const subPositions = metadata?.positions[form.position] ?? [form.sub_position];
  const leagues = metadata?.leagues ?? { [form.league]: form.league };
  const holdoutLabel = metadata?.holdout_season
    ? `${metadata.holdout_season}/${String(metadata.holdout_season + 1).slice(-2)}`
    : null;

  return (
    <div className="mx-auto max-w-5xl px-4 py-8 sm:py-12">
      <header className="mb-8">
        <h1 className="text-2xl font-bold tracking-tight sm:text-3xl">Transfer Value Predictor</h1>
        <p className="mt-2 max-w-2xl text-sm text-slate-600 dark:text-slate-400">
          Estimate a footballer&rsquo;s market value from one season in Europe&rsquo;s top five
          leagues. Change any number and the estimate updates.
        </p>
      </header>

      <div className="grid gap-6 lg:grid-cols-5">
        <form
          className="space-y-6 rounded-2xl border border-slate-200 bg-white p-5 shadow-sm lg:col-span-3 dark:border-slate-800 dark:bg-slate-900"
          onSubmit={(event) => event.preventDefault()}
        >
          <div>
            <span className="mb-2 block text-xs font-medium text-slate-600 dark:text-slate-400">
              Start from an example
            </span>
            <div className="flex flex-wrap gap-2">
              {PRESETS.map((preset) => (
                <button
                  key={preset.label}
                  type="button"
                  onClick={() => setForm(preset.form)}
                  className="rounded-full border border-slate-300 px-3 py-1 text-xs font-medium hover:border-emerald-600 hover:text-emerald-700 dark:border-slate-700 dark:hover:text-emerald-400"
                >
                  {preset.label}
                </button>
              ))}
            </div>
          </div>

          <Section title="Player">
            <div className="grid grid-cols-2 gap-1 rounded-lg bg-slate-100 p-1 sm:grid-cols-4 dark:bg-slate-800">
              {positions.map((position) => (
                <button
                  key={position}
                  type="button"
                  aria-pressed={form.position === position}
                  onClick={() =>
                    update({
                      position,
                      sub_position: metadata?.positions[position][0] ?? form.sub_position,
                    })
                  }
                  className={
                    "rounded-md px-2 py-1.5 text-sm font-medium " +
                    (form.position === position
                      ? "bg-white text-emerald-700 shadow-sm dark:bg-slate-950 dark:text-emerald-400"
                      : "text-slate-600 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100")
                  }
                >
                  {position}
                </button>
              ))}
            </div>
            <div className="grid gap-3 sm:grid-cols-2">
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-400">
                  Detailed position
                </span>
                <select
                  className={inputClass}
                  value={form.sub_position}
                  onChange={(event) => update({ sub_position: event.target.value })}
                >
                  {subPositions.map((name) => (
                    <option key={name}>{name}</option>
                  ))}
                </select>
              </label>
              <label className="block">
                <span className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-400">
                  League
                </span>
                <select
                  className={inputClass}
                  value={form.league}
                  onChange={(event) => update({ league: event.target.value })}
                >
                  {Object.entries(leagues).map(([code, name]) => (
                    <option key={code} value={code}>
                      {name}
                    </option>
                  ))}
                </select>
              </label>
            </div>
            <label className="block">
              <span className="mb-1 flex justify-between text-xs font-medium text-slate-600 dark:text-slate-400">
                <span>Age</span>
                <span className="tabular-nums text-slate-900 dark:text-slate-100">{form.age}</span>
              </span>
              <input
                type="range"
                className="w-full accent-emerald-600"
                min={16}
                max={40}
                value={form.age}
                onChange={(event) => update({ age: Number(event.target.value) })}
              />
            </label>
            <label className="block">
              <span className="mb-1 block text-xs font-medium text-slate-600 dark:text-slate-400">
                Club strength this season
              </span>
              <select
                className={inputClass}
                value={form.clubTier}
                onChange={(event) => update({ clubTier: Number(event.target.value) })}
              >
                {CLUB_TIERS.map((tier, index) => (
                  <option key={tier.label} value={index}>
                    {tier.label} ({tier.points} points per game)
                  </option>
                ))}
              </select>
            </label>
          </Section>

          <Section title="This season (league games)">
            <div className="grid grid-cols-2 gap-3 sm:grid-cols-5">
              <NumberField label="Appearances" value={form.appearances} min={1} max={60}
                onChange={(appearances) => update({ appearances })} />
              <NumberField label="Minutes" value={form.minutes_played} min={1} max={5400}
                onChange={(minutes_played) => update({ minutes_played })} />
              <NumberField label="Goals" value={form.goals} min={0} max={80}
                onChange={(goals) => update({ goals })} />
              <NumberField label="Assists" value={form.assists} min={0} max={60}
                onChange={(assists) => update({ assists })} />
              <NumberField label="Yellow cards" value={form.yellow_cards} min={0} max={25}
                onChange={(yellow_cards) => update({ yellow_cards })} />
            </div>
          </Section>

          <Section title="Last season (leave at 0 if new to these leagues)">
            <div className="grid grid-cols-3 gap-3">
              <NumberField label="Minutes" value={form.prev_minutes} min={0} max={5400}
                onChange={(prev_minutes) => update({ prev_minutes })} />
              <NumberField label="Goals" value={form.prev_goals} min={0} max={80}
                onChange={(prev_goals) => update({ prev_goals })} />
              <NumberField label="Assists" value={form.prev_assists} min={0} max={60}
                onChange={(prev_assists) => update({ prev_assists })} />
            </div>
          </Section>
        </form>

        <aside className="space-y-6 lg:col-span-2">
          <section
            aria-live="polite"
            className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900"
          >
            <h2 className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Estimated market value
            </h2>
            {error ? (
              <p className="mt-3 text-sm text-red-600 dark:text-red-400">
                Couldn&rsquo;t get a prediction: {error}
              </p>
            ) : (
              <>
                <p
                  className={
                    "mt-2 text-5xl font-bold tabular-nums tracking-tight text-emerald-700 transition-opacity dark:text-emerald-400 " +
                    (loading ? "opacity-40" : "")
                  }
                >
                  {prediction ? euros(prediction.predicted_value_eur) : "…"}
                </p>
                {metadata?.holdout_median_pct_error && (
                  <p className="mt-3 text-sm text-slate-600 dark:text-slate-400">
                    Half of the model&rsquo;s predictions land within{" "}
                    {Math.round(metadata.holdout_median_pct_error)}% of the real value, measured on
                    the {holdoutLabel} season it never saw.
                  </p>
                )}
                {prediction && (
                  <p className="mt-1 text-xs text-slate-500">Model: {prediction.model}</p>
                )}
              </>
            )}
          </section>

          {comparables.length > 0 && !error && (
            <section className="rounded-2xl border border-slate-200 bg-white p-5 shadow-sm dark:border-slate-800 dark:bg-slate-900">
              <h2 className="text-sm font-semibold">Real players valued closest</h2>
              <p className="mt-1 text-xs text-slate-500">
                {PLURALS[form.position]}, {holdoutLabel} Transfermarkt values
              </p>
              <ul className="mt-3 divide-y divide-slate-100 dark:divide-slate-800">
                {comparables.map((player) => (
                  <li key={player.name} className="flex items-center justify-between gap-3 py-2">
                    <div className="min-w-0">
                      <p className="truncate text-sm font-medium">{player.name}</p>
                      <p className="text-xs text-slate-500">
                        {leagues[player.league] ?? player.league} · age {Math.floor(player.age)} ·{" "}
                        {player.goals}G {player.assists}A
                      </p>
                    </div>
                    <span className="shrink-0 text-sm font-semibold tabular-nums">
                      {euros(player.market_value_in_eur)}
                    </span>
                  </li>
                ))}
              </ul>
            </section>
          )}
        </aside>
      </div>

      <footer className="mt-8 text-xs text-slate-500">
        Values are Transfermarkt&rsquo;s crowd-sourced estimates, not actual transfer fees. The
        model undervalues superstars, whose worth depends on things statistics don&rsquo;t capture.
      </footer>
    </div>
  );
}
