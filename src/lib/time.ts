/** Durations as a person would say them. */
export function formatDuration(seconds: number): string {
  if (seconds < 60) {
    const whole = Math.max(Math.round(seconds), 1)
    return `${whole} second${whole === 1 ? '' : 's'}`
  }
  const minutes = Math.round(seconds / 60)
  return `${minutes} minute${minutes === 1 ? '' : 's'}`
}

/** Seconds left at the pace so far, or null before there is any pace to go on. */
export function remainingSeconds(progress: number, elapsed: number): number | null {
  if (progress <= 0) return null
  return Math.max(Math.round((elapsed / progress) * (1 - progress)), 0)
}
