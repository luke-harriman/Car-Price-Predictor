import tensorflow as tf


class RSquared(tf.keras.metrics.Metric):
    def __init__(self, name="r_squared", **kwargs):
        super(RSquared, self).__init__(name=name, **kwargs)
        self.SS_res = self.add_weight(name="SS_res", initializer="zeros")
        self.SS_tot = self.add_weight(name="SS_tot", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        ss_res = tf.reduce_sum(tf.square(y_true - y_pred))
        ss_tot = tf.reduce_sum(tf.square(y_true - tf.reduce_mean(y_true)))

        self.SS_res.assign_add(ss_res)
        self.SS_tot.assign_add(ss_tot)
        # Explicitly casting to float32
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return (1 - self.SS_res / (self.SS_tot + tf.keras.backend.epsilon()))


class WeightedAverageInaccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="weighted_average_inaccuracy", **kwargs):
        super(WeightedAverageInaccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        epsilon = 1e-7  # small constant to prevent division by zero
        accuracy = tf.abs((y_pred - y_true) / (y_true + epsilon)) * 100
        weights = y_true / (tf.reduce_sum(y_true) + epsilon)
        weighted_accuracy = tf.reduce_sum(weights * accuracy)
        self.total.assign_add(weighted_accuracy)
        self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))

    def result(self):
        return self.total / self.count



class AverageInaccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="average_inaccuracy", **kwargs):
        super(AverageInaccuracy, self).__init__(name=name, **kwargs)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        inaccuracy = tf.abs((y_pred - y_true) / y_true) * 100
        self.total.assign_add(tf.reduce_sum(inaccuracy))
        # Cast the count of elements to float32
        self.count.assign_add(tf.cast(tf.size(inaccuracy), tf.float32))

    def result(self):
        return self.total / self.count