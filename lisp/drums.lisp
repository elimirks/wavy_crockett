(require "lisp/common")
(require "lisp/wd")

;(wd-plot (wd-amplify 1.0 result))
;(wd-play (wd-amplify 1.0 result))
;(wd-plot (wd-bezier '((0.0 . 0.0) (0.5 . 1.0) (1.0 . 0.0)) wd-full-note-duration))
(wd-plot (wd-slope-down wd-full-note-duration))
