from trainer import Trainer
import multiprocessing


def main():
    model_names = ["A2C", "PPO", "SAC", "TD3"]
    timesteps = 2e6
    contact_reward = 500
    for model_name in model_names:
        trainer = Trainer(
            env_name="AM_Robotics-v0",
            model_name=model_name,
            policy="MlpPolicy",
            n_cpu=int(multiprocessing.cpu_count() * 0.9),
            total_timesteps=timesteps,
            tb_log_name="logs/{}".format(model_name.lower()),
            mjcf_path="/home/root2/AM_Robotics/AM_Robot/arm_target.xml",
            render_mode="human",
            frame_skip=5,
            contact_reward=contact_reward,
            max_steps=1000,
            seed=0,
            model_path="models/{}_timesteps_{}_contact_reward_{}".format(model_name.lower(), timesteps, contact_reward),
        )
        try:
            trainer.train()
            trainer.save()
        except Exception as e:
            print(e)
            trainer.save()
            continue
        except KeyboardInterrupt:
            print("KeyboardInterrupt")
            trainer.save()
            continue


if __name__ == "__main__":
    main()
