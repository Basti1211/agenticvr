import { formatMilliseconds } from "../../utils/retrievalLoop";

type TimingSummaryProps = {
  timings?: Record<string, number>;
};

export const TimingSummary = ({ timings }: TimingSummaryProps) => {
  if (!timings || Object.keys(timings).length === 0) return null;

  return (
    <div className="timing-strip" aria-label="Runtime timings">
      {Object.entries(timings).map(([name, value]) => (
        <span className="timing-pill" key={name}>
          <strong>{name}</strong>
          {formatMilliseconds(value)}
        </span>
      ))}
    </div>
  );
};
