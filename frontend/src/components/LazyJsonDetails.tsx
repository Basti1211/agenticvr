import { useState } from "react";
import type { SyntheticEvent } from "react";

type LazyJsonDetailsProps = {
  className?: string;
  summary: string;
  value: unknown;
};

export const LazyJsonDetails = ({
  className,
  summary,
  value,
}: LazyJsonDetailsProps) => {
  const [isOpen, setIsOpen] = useState(false);

  const handleToggle = (event: SyntheticEvent<HTMLDetailsElement>) => {
    setIsOpen(event.currentTarget.open);
  };

  return (
    <details className={className} onToggle={handleToggle} open={isOpen}>
      <summary>{summary}</summary>
      {isOpen ? <pre>{JSON.stringify(value, null, 2) ?? String(value)}</pre> : null}
    </details>
  );
};
