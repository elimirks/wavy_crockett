(require "lisp/common")
(require "lisp/wd")

(wd-set-bpm 100.0)

(set 'duration (* 8 wd-full-note-duration))

(defun saw (frequency)
  (wd-saw frequency duration))

;; (set 'result
;;      (reduce (lambda (acc it) (wd-concat (wd-concat acc (wd-zero wd-full-note-duration)) it))
;;              (list
;;                ;; using lfo to create a wobble bass
;;                (envelope-lfo 0.5 0.1 5.0 (wd-square g2 duration))
;;                ;; various slight overtones also make cool effects
;;                (wd-amplify 0.25 (wd-superimpose (wd-square g2 duration) (wd-square (* 1.001 g2) duration)))
;;                (wd-amplify 0.25 (wd-superimpose (wd-square g2 duration) (wd-square (* 1.005 g2) duration)))
;;                (wd-amplify 0.25 (wd-superimpose (wd-square g2 duration) (wd-square (* 1.01 g2) duration)))
;;                ;; make a chord by superimposing a multiple of two
;;                (wd-amplify 0.25 (wd-chord g2 (list 2.0) saw))
;;                ;; or another chord, by increments of 3/2 multiples
;;                (wd-amplify 0.25 (wd-chord g2 (list 1.5 2.0) saw))
;;                ;; anotha one
;;                (wd-amplify 0.25 (wd-chord g2 (list 2.0 3.0) saw))
;;                (wd-amplify 0.25 (wd-chord g2 (list 1.25 2.0) saw))
;;                (wd-amplify 0.25 (wd-chord g2 (list 1.25 1.5) saw)))))
;; (wd-play result)
;(wd-save result "so_i_heard_ya_like_overtones.wav")

(wd-play (envelope-soft-hold (wd-triangle g3 wd-full-note-duration)))
(wd-play (envelope-soft-hold (wd-gaussian-blur 1.0 (wd-triangle g3 wd-full-note-duration))))
(wd-play (envelope-soft-hold (wd-gaussian-blur 2.0 (wd-triangle g3 wd-full-note-duration))))
(wd-play (envelope-soft-hold (wd-gaussian-blur 3.0 (wd-triangle g3 wd-full-note-duration))))
(wd-play (envelope-soft-hold (wd-gaussian-blur 4.0 (wd-triangle g3 wd-full-note-duration))))
(wd-play (envelope-soft-hold (wd-gaussian-blur 5.0 (wd-triangle g3 wd-full-note-duration))))
(wd-play (envelope-soft-hold (wd-gaussian-blur 6.0 (wd-triangle g3 wd-full-note-duration))))
(wd-play (envelope-soft-hold (wd-gaussian-blur 7.0 (wd-triangle g3 wd-full-note-duration))))
(wd-play (envelope-soft-hold (wd-gaussian-blur 8.0 (wd-triangle g3 wd-full-note-duration))))
(wd-play (envelope-soft-hold (wd-gaussian-blur 9.0 (wd-triangle g3 wd-full-note-duration))))
