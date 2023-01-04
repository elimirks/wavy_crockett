(require "lisp/common")
(require "lisp/wd")

(wd-set-bpm 100.0)
(set 'drum-loop (wd-build-track
  (list
    (list (synth-kick))
    nil
    (list (wd-amplify 0.7 (synth-noise-hat-muted)))
    nil
    (list (synth-kick))
    nil
    (list (wd-amplify 0.5 (synth-noise-hat)))
    nil
    (list (synth-kick))
    nil
    (list (wd-amplify 0.5 (synth-noise-hat-muted)))
    nil
    (list (wd-amplify 0.7 (synth-kick))))))

(wd-play (reduce wd-concat
                 (list drum-loop
                       drum-loop
                       drum-loop
                       drum-loop)))
