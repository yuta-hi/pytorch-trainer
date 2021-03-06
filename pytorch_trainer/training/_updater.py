class Updater(object):

    """Interface of updater objects for trainers.

    :class:`~pytorch_trainer.training.Updater` implements a training iteration
    as :meth:`update`. Typically, the updating iteration proceeds as follows.

    - Fetch a minibatch from :mod:`~pytorch_trainer.dataset`
      via :class:`~pytorch_trainer.dataset.Iterator`.
    - Run forward and backward process of :class:`~pytorch_trainer.Chain`.
    - Update parameters according to their :class:`~pytorch_trainer.UpdateRule`.

    The first line is processed by
    :meth:`Iterator.__next__ <pytorch_trainer.dataset.Iterator.__next__>`.
    The second and third are processed by
    :meth:`Optimizer.update <pytorch_trainer.Optimizer.update>`.
    Users can also implement their original updating iteration by overriding
    :meth:`Updater.update <pytorch_trainer.training.Updater.update>`.

    """

    def connect_trainer(self, trainer):
        """Connects the updater to the trainer that will call it.

        The typical usage of this method is to register additional links to the
        reporter of the trainer. This method is called at the end of the
        initialization of :class:`~pytorch_trainer.training.Trainer`. The default
        implementation does nothing.

        Args:
            trainer (~pytorch_trainer.training.Trainer): Trainer object to which the
                updater is registered.

        """
        pass

    def finalize(self):
        """Finalizes the updater object.

        This method is called at the end of training loops. It should finalize
        each dataset iterator used in this updater.

        """
        raise NotImplementedError

    def get_optimizer(self, name):
        """Gets the optimizer of given name.

        Updater holds one or more optimizers with names. They can be retrieved
        by this method.

        Args:
            name (str): Name of the optimizer.

        Returns:
            ~pytorch_trainer.Optimizer: Optimizer of the name.

        """
        raise NotImplementedError

    def get_all_optimizers(self):
        """Gets a dictionary of all optimizers for this updater.

        Returns:
            dict: Dictionary that maps names to optimizers.

        """
        raise NotImplementedError

    def get_all_models(self):
        """Gets a dictionary of all models for this updater.

        Returns:
            dict: Dictionary that maps names to models.

        """
        raise NotImplementedError

    def update(self):
        """Updates the parameters of the target model.

        This method implements an update formula for the training task,
        including data loading, forward/backward computations, and actual
        updates of parameters.

        This method is called once at each iteration of the training loop.

        """
        raise NotImplementedError

    def state_dict(self):
        """Serializes the current state of the updater object."""
        raise NotImplementedError

    def load_state_dict(self, state_dict):
        raise NotImplementedError
