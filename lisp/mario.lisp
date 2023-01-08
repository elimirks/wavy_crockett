(require "lisp/common")
(require "lisp/wd")

(wd-set-bpm 167)

;; Try playing with the cardinality parameters and changing which
;; chord multiplier set the `mario-melody` function uses.
(set 'wd-chord-multipliers-1 (wd-create-chord-multipliers 1.0 2.0 2))
(set 'wd-chord-multipliers-2 (wd-create-chord-multipliers 2.0 2.0 2))
(set 'wd-chord-multipliers-3 (wd-create-chord-multipliers 3.0 2.0 2))
(set 'wd-chord-multipliers-4 (wd-create-chord-multipliers 4.0 2.0 2))
(set 'wd-chord-multipliers-5 (wd-create-chord-multipliers 1.5 1.0 2))
(set 'wd-chord-multipliers-6 (wd-create-chord-multipliers 2.0 1.0 2))
(set 'wd-chord-multipliers-7 (wd-create-chord-multipliers (/ 5.0 3.0) (/ 2.0 3.0) 2))

(defun mario-melody (frequency)
  (wd-amplify 0.1 (wd-chord frequency wd-chord-multipliers-8 synth-pure)))

(set 'mario-intro
     (wd-build-track
       (list
         (list (envelope-soft-quarter (mario-melody e4)))
         nil
         (list (envelope-soft-quarter (mario-melody e4)))
         nil
         nil
         (list (envelope-soft-half (mario-melody e4)))
         nil
         nil
         (list (envelope-soft-quarter (mario-melody c4)))
         nil
         (list (envelope-soft-quarter (mario-melody e4)))
         nil
         nil
         (list (envelope-soft-half (mario-melody g4)))
         nil
         nil
         nil
         nil
         nil
         (list (envelope-soft-half (mario-melody g3)))
         (list (wd-zero (* 6 wd-quarter-note-duration))))))

(set 'mario-body-1
     (wd-build-track
       (list
         (list (envelope-soft-half (mario-melody c4)))
         nil
         nil
         nil
         nil
         (list (envelope-soft-half (mario-melody g3)))
         nil
         nil
         nil
         nil
         (list (envelope-soft-half (mario-melody e3)))
         nil
         nil
         nil
         (list (envelope-soft-quarter (mario-melody a3)))
         nil
         (list (envelope-soft-half (mario-melody b3)))
         nil
         nil
         nil
         (list (envelope-soft-quarter (mario-melody as3)))
         nil
         (list (envelope-soft-half (mario-melody a3)))
         nil
         nil
         (list (envelope-soft-half (mario-melody g3)))
         nil
         nil
         nil
         (list (envelope-soft-quarter (mario-melody e4)))
         nil
         (list (envelope-soft-quarter (mario-melody g4)))
         nil
         (list (envelope-soft-half (mario-melody a4)))
         nil
         nil
         nil
         (list (envelope-soft-quarter (mario-melody f4)))
         nil
         (list (envelope-soft-quarter (mario-melody g4)))
         nil
         nil
         (list (envelope-soft-quarter (mario-melody e4)))
         nil
         (list (envelope-soft-quarter (mario-melody c4)))
         nil
         (list (envelope-soft-quarter (mario-melody d4)))
         nil
         (list (envelope-soft-quarter (mario-melody b3)))
         (list (wd-zero (* 1 wd-full-note-duration))))))

;; (set 'mario-body-header-1
;;      (wd-build-track
;;        (list
;;          (list (envelope-soft-quarter (mario-melody g4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody fs4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody f4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody d4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody e4)))
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody g3)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody a3)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody a3)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody d4)))
;;          (list (wd-zero (* 1 wd-full-note-duration))))))
;;
;; (set 'mario-body-2
;;      (wd-build-track
;;        (list
;;          (list (envelope-soft-quarter (mario-melody g4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody fs4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody f4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody d4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody e4)))
;;          nil
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c5)))
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c5)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c5)))
;;          (list (wd-zero (* 1 wd-full-note-duration))))))
;;
;; (set 'mario-body-3
;;      (wd-build-track
;;        (list
;;          (list (envelope-soft-half (mario-melody ds4)))
;;          nil
;;          nil
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody d4)))
;;          nil
;;          nil
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody c4)))
;;          (list (wd-zero (* 1 wd-full-note-duration))))))
;;
;; (set 'mario-body-header-2
;;      (wd-build-track
;;        (list
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody d4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody e4)))
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody a3)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody g3)))
;;          (list (wd-zero (* 1 wd-full-note-duration))))))
;;
;; (set 'mario-body-4
;;      (wd-build-track
;;        (list
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody d4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody e4)))
;;          (list (wd-zero (* 2 wd-full-note-duration))))))
;;
;; (set 'mario-body-header-3
;;      (wd-build-track
;;        (list
;;          (list (envelope-soft-quarter (mario-melody e4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody g4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody g4)))
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody a3)))
;;          nil
;;          nil
;;          (list (envelope-soft-quarter (mario-melody f4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody f4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody a3)))
;;          (list (wd-zero (* 1 wd-full-note-duration))))))
;;
;; (set 'mario-body-5
;;      (wd-build-track
;;        (list
;;          (list (envelope-soft-quarter (mario-melody b3)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody a4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody a4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody a4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody g4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody f4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody e4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody a3)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody g3)))
;;          (list (wd-zero (* 1 wd-full-note-duration))))))
;;
;; (set 'mario-body-6
;;      (wd-build-track
;;        (list
;;          (list (envelope-soft-quarter (mario-melody b3)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody f4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody f4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody f4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody e4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody d4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c4)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody g3)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody e3)))
;;          nil
;;          (list (envelope-soft-quarter (mario-melody c3)))
;;          (list (wd-zero (* 1 wd-full-note-duration))))))
;;
;; (set 'mario-outro
;;      (wd-build-track
;;        (list
;;          (list (envelope-soft-half (mario-melody c4)))
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody g3)))
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody e3)))
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody a3)))
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody b3)))
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody a3)))
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody gs3)))
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody as3)))
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody gs3)))
;;          nil
;;          nil
;;          (list (envelope-soft-half (mario-melody g3)))
;;          (list (envelope-soft-half (mario-melody fs3)))
;;          nil
;;          (list (envelope-soft-half (mario-melody g3)))
;;          nil
;;          nil
;;          (list (wd-zero (* 1 wd-full-note-duration))))))

(set 'result (reduce wd-concat
                     (list mario-intro
                           mario-body-1
                           mario-body-1
                           ;; mario-body-header-1
                           ;; mario-body-2
                           ;; mario-body-header-1
                           ;; mario-body-3
                           ;; mario-body-header-2
                           ;; mario-body-4
                           ;; mario-body-header-2
                           ;; mario-intro
                           ;; mario-body-1
                           ;; mario-body-1
                           ;; mario-body-header-3
                           ;; mario-body-5
                           ;; mario-body-header-3
                           ;; mario-body-6
                           ;; mario-outro
                           )))

(wd-play (wd-amplify 0.5 result))
;(wd-save (wd-amplify 0.5 result) "mario.wav")
