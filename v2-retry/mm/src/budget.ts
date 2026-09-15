/**
 * Graceful failure for walks that grow past what the machine can hold. Machine side only: a wall-clock deadline
 * checked every so many steps. When it passes, the walk stops with the stage it had reached; nothing is written as
 * if the cycle were complete.
 */

export class WalkBudgetExceeded extends Error {
  readonly stage: string;
  readonly detail: string;
  constructor(stage: string, detail: string) {
    super(`walk budget exceeded at ${stage}${detail ? ` (${detail})` : ""}`);
    this.stage = stage;
    this.detail = detail;
  }
}

/** called inside long loops: `stage` names where the walk is; `detail` is only built when the budget has run out */
export type Tick = (stage: string, detail?: () => string) => void;

export const noTick: Tick = () => {};

/** a tick that throws once `deadline` (ms since epoch) has passed; `onStage` hears each stage the first time it is entered */
export function deadlineTick(deadline: number, onStage?: (stage: string) => void): Tick {
  let steps = 0;
  let last = "";
  return (stage, detail) => {
    if (stage !== last) {
      last = stage;
      onStage?.(stage);
    }
    if ((++steps & 255) === 0 && Date.now() > deadline) throw new WalkBudgetExceeded(stage, detail?.() ?? "");
  };
}
