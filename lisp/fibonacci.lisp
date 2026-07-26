(require "lisp/common")
(require "lisp/wd")

(set 'fib (fold (lambda (acc it)
        (if (le? it 1)
            (push 1 acc)
            (push (+ (nth (- it 1) acc) (nth (- it 2) acc)) acc)))
      '()
      (range 0 50)))

(wd-set-bpm 120.0)

(defun synth-round (n)
  (if (eq 0 n)
    (wd-pure-tone 0 wd-full-note-duration)
    (progn 
      (set 'base (wd-pure-tone (nth n fib) wd-full-note-duration))
      (set 'duration wd-full-note-duration)
      (set 'attack (/ duration 32))
      (set 'decay (/ duration 2))
      (set 'sustain 0.7)
      (set 'release (/ duration 16))
      (wd-pad-to-full-note (wd-adsr attack decay sustain release base)))))

(defun triangle-high (n)
  (if (eq 0 n)
    (wd-pure-tone 0 wd-full-note-duration)
    (progn
      (set 'base (wd-triangle (nth n fib) wd-full-note-duration))
      (set 'duration wd-full-note-duration)
      (set 'attack (/ duration 10))
      (set 'decay (/ duration 5))
      (set 'sustain 0.30)
      (set 'release (/ duration 5))
      (wd-pad-to-full-note (wd-adsr attack decay sustain release base)))))
;
; (wd-play (triangle-high 16))
; (wd-play (triangle-high 15))
; (wd-play (triangle-high 17))
; (wd-play (triangle-high 14))
;
; (exit 0)

(set 'triangle-sequence
     (list
       0 0 0 0
       0 0 0 0
       0 15 16 14
       16 15 17 0

       0 0 0 0
       0 0 13 12
       0 0 0 0
       0 0 12 10

       0 0 0 0
       0 0 0 18
       0 0 0 0
       0 0 12 0
))

(set 'round-sequence
     (list
       9 10 11 9
       9 10 11 9
       9 10 11 9
       9 10 11 0

       13 12 13 12
       11 12 11 12
       10 11 10 11
       11 10 11 0

       9 10 11 9
       9 10 11 0
       9 10 11 9
       9 10 11 9
))


(set 'drum-loop (reduce wd-concat (list
    (synth-kick)
    (wd-amplify 0.4 (synth-noise-hat))
    (wd-amplify 0.3 (synth-noise-hat))
    (wd-amplify 0.4 (synth-noise-hat-muted)))))

(set 'round-wd (wd-gaussian-blur 0.6
                    (reduce wd-concat (map synth-round round-sequence))))
(set 'triangle-wd (reduce wd-concat (map triangle-high triangle-sequence)))

(set 'drum-loop-count (/ (wd-len round-wd) (wd-len drum-loop)))
(set 'drums (wd-repeat drum-loop-count drum-loop))

(set 'song (reduce wd-superimpose (list
             (wd-amplify 0.3 drums)
             (wd-amplify 0.8 round-wd)
             (wd-amplify 0.3 triangle-wd))))

; (wd-play song)
(wd-save song "fib.wav")
