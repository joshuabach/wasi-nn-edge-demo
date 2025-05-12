;;; Directory Local Variables
;;; For more information see (info "(emacs) Directory Variables")

((lsp-mode . ((lsp-rust-analyzer-cargo-watch-args . ["check"
                                                     (\, "--message-format=json")])
              (lsp-rust-analyzer-cargo-watch-command . "component")
              (lsp-rust-analyzer-cargo-override-command . ["cargo"
                                                           (\, "check")
                                                           (\, "--workspace")
                                                           (\, "--target=wasm32-wasip2")
                                                           (\, "--message-format=json")]))))
