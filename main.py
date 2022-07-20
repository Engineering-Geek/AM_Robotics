from trainer import Trainer
import multiprocessing


def main():
    for model in ["PPO", "A2C", "DQN", "SAC", "TD3"]:
        trainer = Trainer(
            env_name="AM_Robotics-v0",
            model_name=model,
            policy="MlpPolicy",
            n_cpu=int(multiprocessing.cpu_count() * 0.8),
            total_timesteps=1e6,
            tb_log_name="logs/{}".format(model),
            mjcf_path="/home/nmelgiri/AM_Robotics/AM_Robot/arm_target.xml",
            render_mode="human",
            frame_skip=5,
            contact_reward=150,
            max_steps=1000,
            seed=0,
            model_path='models' + model.lower() + "_arm"
        )
        trainer.train()
        trainer.close()


if __name__ == "__main__":
    main()
