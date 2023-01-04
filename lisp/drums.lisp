(require "lisp/common")
(require "lisp/wd")

;(wd-plot (wd-from-frequencies (wd-amplify d2 (wd-slope-down wd-full-note-duration))))
(wd-plot (wd-shifting-pure-tone d2 0.0 (* 80 wd-full-note-duration)))
