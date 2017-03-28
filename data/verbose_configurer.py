

class VerboseConfigurer:
    def _apply_verbose(self, verbose):
        if hasattr(self, 'transformer_list'):
            transformer_list = self.transformer_list
        elif hasattr(self, 'steps'):
            transformer_list = self.steps
        else:
            self.verbose = verbose
            return

        for name, transformer in transformer_list:
            if hasattr(transformer, '_apply_verbose'):
                transformer._apply_verbose(verbose)
            elif hasattr(transformer, 'verbose'):
                transformer.verbose = verbose
