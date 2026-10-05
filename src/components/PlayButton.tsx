interface Props {
  playing: boolean
  onClick: () => void
}

export function PlayButton({ playing, onClick }: Props) {
  return (
    <button type="button" className="play" onClick={onClick}>
      <svg className="play-icon" viewBox="0 0 16 16" aria-hidden="true">
        {playing
          ? <path d="M3 2h3.5v12H3zM9.5 2H13v12H9.5z" />
          : <path d="M4 1.8v12.4a.6.6 0 0 0 .92.5l9.5-6.2a.6.6 0 0 0 0-1L4.92 1.3a.6.6 0 0 0-.92.5z" />}
      </svg>
      {playing ? 'Pause' : 'Play'}
    </button>
  )
}
