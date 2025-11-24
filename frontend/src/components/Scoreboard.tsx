import type { GameStateView } from '../types';
import { getPlayerColor } from '../utils/iconMapper';

const SCORE_CATEGORY_ORDER = ['territory', 'population', 'military', 'economy'] as const;

const SCORE_CATEGORY_LABELS: Record<string, string> = {
  territory: 'Territoire',
  population: 'Population',
  military: 'Militaire',
  economy: 'Économie',
};

const NUMBER_FORMATTER = new Intl.NumberFormat('fr-FR');

const formatValue = (value?: number) =>
  typeof value === 'number' ? NUMBER_FORMATTER.format(value) : '—';

const safeLength = (arr?: number[]) => (Array.isArray(arr) ? arr.length : 0);

interface ScoreboardProps {
  state: GameStateView;
  highlightPlayer?: number;
  title?: string;
  className?: string;
}

interface ScoreRow {
  playerId: number;
  score?: number;
  stars?: number;
  income?: number;
  breakdown: Record<string, number | undefined>;
}

export function Scoreboard({
  state,
  highlightPlayer,
  title = 'Classement',
  className,
}: ScoreboardProps) {
  const breakdown = state.score_breakdown ?? {};

  const canonicalCategories = SCORE_CATEGORY_ORDER.filter(
    (key) => Array.isArray(breakdown[key]) && breakdown[key].length > 0
  );
  const extraCategories = Object.keys(breakdown)
    .filter((key) => !SCORE_CATEGORY_ORDER.includes(key as typeof SCORE_CATEGORY_ORDER[number]))
    .filter((key) => Array.isArray(breakdown[key]) && breakdown[key].length > 0)
    .sort();
  const categoryKeys = [...canonicalCategories, ...extraCategories];

  const lengthCandidates = [
    safeLength(state.player_score),
    safeLength(state.player_stars),
    safeLength(state.player_income),
    ...categoryKeys.map((key) => safeLength(breakdown[key])),
  ].filter((value) => value > 0);
  const playerCount = lengthCandidates.length > 0 ? Math.max(...lengthCandidates) : 0;

  if (playerCount === 0) {
    return (
      <div
        className={`bg-gray-900/60 rounded-2xl p-4 text-sm text-gray-400 ${
          className ?? ''
        }`.trim()}
      >
        <p className="uppercase tracking-wide text-[11px] text-gray-500">{title}</p>
        <p className="mt-2">Aucun score disponible pour le moment.</p>
      </div>
    );
  }

  const rows: ScoreRow[] = Array.from({ length: playerCount }, (_, playerId) => {
    const rowBreakdown: Record<string, number | undefined> = {};
    categoryKeys.forEach((key) => {
      const values = breakdown[key];
      rowBreakdown[key] = Array.isArray(values) ? values[playerId] : undefined;
    });

    return {
      playerId,
      score: state.player_score?.[playerId],
      stars: state.player_stars?.[playerId],
      income: state.player_income?.[playerId],
      breakdown: rowBreakdown,
    };
  });

  const safeSortValue = (value?: number) =>
    typeof value === 'number' ? value : Number.NEGATIVE_INFINITY;

  const sortedRows = [...rows].sort(
    (a, b) => safeSortValue(b.score) - safeSortValue(a.score)
  );

  return (
    <div
      className={`bg-gray-900/60 rounded-2xl p-4 space-y-3 text-white ${
        className ?? ''
      }`.trim()}
    >
      <div className="flex items-center justify-between">
        <p className="uppercase tracking-widest text-[11px] text-gray-400">{title}</p>
        <p className="text-xs text-gray-500">
          {playerCount} {playerCount === 1 ? 'tribu' : 'tribus'}
        </p>
      </div>

      <div className="space-y-2">
        {sortedRows.map((row) => {
          const isHighlighted = highlightPlayer === row.playerId;
          const rowClasses = isHighlighted
            ? 'border border-emerald-400/40 bg-emerald-500/10'
            : 'border border-gray-700 bg-white/5';
          return (
            <div
              key={`score-row-${row.playerId}`}
              className={`rounded-2xl p-3 ${rowClasses}`}
            >
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-2">
                  <span
                    className="w-3 h-3 rounded-full border border-white/30"
                    style={{ backgroundColor: getPlayerColor(row.playerId) }}
                  />
                  <div className="flex items-center gap-2 text-sm font-semibold">
                    <span>Joueur {row.playerId + 1}</span>
                    {isHighlighted && (
                      <span className="px-2 py-0.5 text-[10px] uppercase tracking-wide text-emerald-200 bg-emerald-900/80 rounded-full">
                        Vous
                      </span>
                    )}
                  </div>
                </div>
                <div className="text-2xl font-bold">
                  {formatValue(row.score)}
                </div>
              </div>

              <div className="mt-2 flex flex-wrap items-center gap-3 text-xs text-gray-300">
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-gray-500">
                    Étoiles
                  </p>
                  <p className="font-semibold">{formatValue(row.stars)} ★</p>
                </div>
                <div>
                  <p className="text-[10px] uppercase tracking-wide text-gray-500">
                    Revenu
                  </p>
                  <p className="font-semibold">
                    {formatValue(row.income)} ★/tour
                  </p>
                </div>
              </div>

              {categoryKeys.length > 0 && (
                <div className="mt-3 flex flex-wrap gap-2 text-[11px] text-gray-200">
                  {categoryKeys.map((key) => (
                    <span
                      key={`${row.playerId}-${key}`}
                      className="rounded-full bg-white/5 px-3 py-1"
                    >
                      <span className="text-xs uppercase tracking-wide text-gray-400">
                        {SCORE_CATEGORY_LABELS[key] ?? key}
                      </span>
                      <span className="ml-2 font-semibold">
                        {formatValue(row.breakdown[key])}
                      </span>
                    </span>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
