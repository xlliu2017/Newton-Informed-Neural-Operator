import unittest

from training_cli import build_training_configs, parse_training_args


class TrainingCliTests(unittest.TestCase):
    def test_defaults_match_single_solution_profile(self):
        args = parse_training_args(
            epochs=1000,
            batch_size=50,
            lr=2e-4,
            loss_type="pde",
            model_type_help="model type",
            argv=[],
        )

        self.assertEqual(args["epochs"], 1000)
        self.assertEqual(args["batch_size"], 50)
        self.assertEqual(args["lr"], 2e-4)
        self.assertEqual(args["loss_type"], "pde")
        self.assertEqual(args["num_iteration"], [[1, 0], [1, 0], [1, 0], [1, 1], [2, 0]])

    def test_num_iteration_and_sampling_rate_are_normalized(self):
        args = parse_training_args(
            epochs=500,
            batch_size=10,
            lr=2e-4,
            loss_type="pde",
            model_type_help="model type",
            argv=[
                "--data",
                "darcy",
                "--sample_x",
                "--num_iteration",
                "[1,0]",
                "[2,1]",
            ],
        )

        self.assertEqual(args["sampling_rate"], 2)
        self.assertEqual(args["num_iteration"], [[1, 0], [2, 1]])

    def test_build_training_configs_creates_expected_sections(self):
        args = parse_training_args(
            epochs=1000,
            batch_size=50,
            lr=1e-4,
            loss_type="l2",
            model_type_help="model type, MgNO_DC_5, MgNO_DC_6, FNO, DeepONet",
            argv=["--normalizer", "--MODEL_PATH_LOAD", "checkpoint.pt"],
        )

        data_options, model_options, optimizer_options = build_training_configs(
            args, lambda options: {**options, "grid_size": 63}
        )

        self.assertEqual(data_options["grid_size"], 63)
        self.assertEqual(data_options["MODEL_PATH_LOAD"], "checkpoint.pt")
        self.assertTrue(model_options["normalizer"])
        self.assertEqual(model_options["activation"], "gelu")
        self.assertEqual(optimizer_options["epochs"], 1000)
        self.assertEqual(optimizer_options["lr"], 1e-4)


if __name__ == "__main__":
    unittest.main()
