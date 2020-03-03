using System;
using System.Collections.Generic;
using System.Linq;
using Keras;
using Keras.Layers;
using Keras.Models;
using Keras.Optimizers;
using Numpy;
using Python.Runtime;
using Tensorflow;
using np = Numpy.np;

namespace Practical_Tensorflow.Sequences
{
    public class Test
    {
        public static Test operator /(Test a, float f)
        {
            return new Test();
        }
        public static Test operator /(float f, Test a)
        {
            return new Test();
        }
    }

    public class Week2
    {
        static void plot_series(NDarray time, NDarray series, string format= "-")
        {
            using var gil = Py.GIL();
            dynamic mpl = Py.Import("matplotlib");
            //mpl.use("TkAgg");
            dynamic plt = Py.Import("matplotlib.pyplot");

            plt.plot(time.PyObject, series.PyObject, format);
            plt.xlabel("Time");
            plt.ylabel("Value");
            plt.grid(false);

            plt.show();
        }

        static NDarray Trend(NDarray time, float slope)
        {
            return slope * time;
        }

        private static NDarray seasonal_pattern(NDarray season_time)
        {
            // Just an arbitrary pattern, you can change it if you wish
            return np.@where(season_time < 0.1,
                np.cos(season_time * 6 * np.pi),
                np.full(new Numpy.Models.Shape(season_time.len), 2.0f) / np.exp(9.0 * season_time));
        }

        static NDarray Seasonality(NDarray time, int period, float amplitude = 1, float phase=0)
        {
            // Repeats the same pattern at each period
            var seasonTime = ((time + phase) % period) / period;
            return seasonal_pattern(seasonTime) * amplitude;
        }

        static NDarray noise(NDarray time, int noise_level = 1, int? seed = 0)
        {
            np.random.RandomState(seed);
            return np.random.randn(time.len) * noise_level;
        }

        private static NDarray windowed_dataset(NDarray series, int window_size, int batch_size, int shuffle_buffer)
        {
            var rows = new List<NDarray>(); // a python list to hold the windows

            // having series: [1, 2, 3, 4..]
            // make sliding window 2D set
            for (int i = 0; i < series.shape[0] - window_size + 1; i++)
            {
                // Split each row into features [1..N-1] and label [N]
                var features = series[$"{i}:{i + window_size-1}"]; 
                var labels = series[$"{i + window_size-1}:{i + window_size}"];
                var featureAndLabel = np.hstack(features, labels); 
                rows.Add(featureAndLabel);
            }

            // rows look now like
            // [
            // [1, 2, 3, 4, 5], [6]
            // [2, 3, 4, 5, 6], [7]
            // [3, 4, 5, 6, 7], [8]
            // [4, 5, 6, 7, 8], [9]
            // ]

            // shuffle 
            var npRows = np.vstack(rows.ToArray());
            np.random.shuffle(npRows);

            // rows look now like
            // [
            // [4, 5, 6, 7, 8], [9]
            // [2, 3, 4, 5, 6], [7]
            // [3, 4, 5, 6, 7], [8]
            // [1, 2, 3, 4, 5], [6]
            // ]

            return npRows;


            // def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
            //     dataset = tf.data.Dataset.from_tensor_slices(series)
            //     dataset = dataset.window(window_size + 1, shift = 1, drop_remainder = True)
            //     dataset = dataset.flat_map(lambda window: window.batch(window_size + 1))
            //     dataset = dataset.shuffle(shuffle_buffer).map(lambda window: (window[:-1], window[-1]))
            //     dataset = dataset.batch(batch_size).prefetch(1)
            //     return dataset
        }
        public static void Run()
        {


            var (time, series) = GenerateData();
            //plot_series(time, series);

            var split_time = 3000;

            // var time_train = time[..split_time];
            // var time_valid = time[split_time..];
            // var x_train = series[..split_time];
            // var x_valid = series[split_time..];

            var time_train = time[$":{split_time}"];
            var time_valid = time[$"{split_time}:"];
            var x_train = series[$":{split_time}"];
            var x_valid = series[$"{split_time}:"];


            var window_size = 20;
            var batch_size = 32;
            var shuffle_buffer_size = 1000;


            var dataset = windowed_dataset(/*x_train*/ np.arange(100), window_size, batch_size, shuffle_buffer_size);

            var hidden1 = new Dense(100, input_shape: new Keras.Shape(window_size), activation: "relu");
            var hidden2 = new Dense(10, activation: "relu");
            var model = new Sequential(new BaseLayer[] {hidden1, hidden2, new Dense(1)});

            model.Compile(loss: "mse", optimizer: new SGD(lr: 1e-6f, momentum:  0.9f));
            //model.Fit(x: dataset, epochs: 100, verbose: 0);

            // //Load train data
            //
            // var x = np.array(new float[,] {{0, 0}, {0, 1}, {1, 0}, {1, 1}});
            // var y = np.array(new float[] {0, 1, 1, 0});
            //
            // //Build functional model
            // var input = new Input(shape: new Shape(2));
            // var hidden1 = new Dense(32, activation: "relu").Set(input);
            // var hidden2 = new Dense(64, activation: "relu").Set(hidden1);
            // var output = new Dense(1, activation: "sigmoid").Set(hidden2);
            // var model = new Model(new BaseLayer[] {input}, new[] {output});
            //
            // //Compile and train
            // model.Compile(optimizer: new Adam(), loss: "binary_crossentropy", metrics: new[] {"accuracy"});
            //
            // var history = model.Fit(x, y, batch_size: 2, epochs: 10);
            // //var weights = model.GetWeights();
            // //model.SetWeights(weights);
            // var logs = history.HistoryLogs;
            // //Save model and weights
            // string json = model.ToJson();
            // File.WriteAllText("model.json", json);
            // model.SaveWeight("model.h5");
            // //Load model and weight
            // var loaded_model = Sequential.ModelFromJson(File.ReadAllText("model.json"));
            // loaded_model.LoadWeight("model.h5");

        }

        private static (NDarray time, NDarray series) GenerateData()
        {
            var time = np.arange(10.0 * 365 + 1);
            var baseline = 10;
            var amplitude = 40;
            var slope = 0.005f;
            var noise_level = 3;
            // Create the series
            var series = baseline + Trend(time, slope) + Seasonality(time, period: 365, amplitude);
            // Update with noise
            series += noise(time, noise_level, seed: 51);
            return (time, series);
        }
    }
}