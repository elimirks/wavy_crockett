(require "lisp/common")
(require "lisp/wd")

(wd-set-bpm 100.0)

; Using LFO to create a wobble bass
(wd-play (envelope-lfo 0.5 0.1 5.0 (wd-square g2 (* 8 wd-full-note-duration))))
; Using LFO to add some spice to a sawtooth wave
(wd-play (envelope-lfo 0.5 0.3 5.0 (wd-saw g3 (* 8 wd-full-note-duration))))
