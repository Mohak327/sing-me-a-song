import { useId } from 'react'

interface Props {
  question: string
  labels: string[]
  chosen: number
  onChoose: (index: number) => void
  disabled?: boolean
}

/** A small set of mutually exclusive options, shown as pills. */
export function Choices({ question, labels, chosen, onChoose, disabled }: Props) {
  const group = useId()
  return (
    <fieldset className="choices-field" disabled={disabled}>
      <legend>{question}</legend>
      <div className="choices">
        {labels.map((label, i) => (
          <label key={label} className={i === chosen ? 'choice chosen' : 'choice'}>
            <input type="radio" name={group} checked={i === chosen} onChange={() => onChoose(i)} />
            {label}
          </label>
        ))}
      </div>
    </fieldset>
  )
}
