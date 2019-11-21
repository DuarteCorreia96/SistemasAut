from ekf import Measurement, EKF_SLAM, MotionModel

def main():

    ekf = EKF_SLAM(MotionModel)
    ekf.print_state()

    ekf.prediction_step(1, 2, 10)
    ekf.print_state()

    measurs = [Measurement(1, 2, 0.4), Measurement(2, 2, 0.25)]
    ekf.update_step(measurs)

if __name__ == "__main__":
    main()