# Параметры запуска dual_crl_ppo.py

Запуск выполняется из каталога jaxgcrl:

~~~bash
python dual_crl_ppo.py [параметры]
~~~

Скрипт не читает конфигурационные файлы: все настройки задаются через CLI.
Ниже перечислены все 56 аргументов из dual_crl_ppo.py. Флаги с префиксом
--no_ не принимают значения: соответствующая возможность включена по умолчанию,
а такой флаг её отключает.

## Быстрые варианты

Full Craftax, semantic goals, без W&B:

~~~bash
python dual_crl_ppo.py --env_name Craftax-Symbolic-v1 --no_use_wandb
~~~

Craftax Classic, observation-level goals:

~~~bash
python dual_crl_ppo.py --env_name Craftax-Classic-Symbolic-v1 --goal_set observation --seed 42 --no_use_wandb
~~~

Gridworld:

~~~bash
python dual_crl_ppo.py --env_name Gridworld-v1 --network_type symbolic_flat --no_use_wandb
~~~

W&B и сохранение PPO policy:

~~~bash
python dual_crl_ppo.py --env_name Craftax-Symbolic-v1 --goal_set observation --seed 42 --total_timesteps 1e10 --wandb_project crl-craftax --save_policy
~~~

## Расчёт объёма обучения

За один большой update собирается rollout размера:

~~~text
rollout_size = num_steps * num_envs
num_updates = total_timesteps // rollout_size
~~~

Фактическое число переходов равно num_updates * num_steps * num_envs и может
быть меньше total_timesteps. Значение total_timesteps также используется для
SPS и как шаг checkpoint-а.

ppo_minibatch_size и crl_minibatch_size должны делить rollout_size без остатка.
При стандартных num_steps=64 и num_envs=1024 размер rollout равен 65536, а
minibatch size 2048 даёт 32 minibatch-а.

## 1. Общие параметры

### --env_name

~~~text
Тип: str
По умолчанию: Craftax-Symbolic-v1
~~~

Выбирает среду:

- Craftax-Symbolic-v1 — Full Craftax;
- Craftax-Classic-Symbolic-v1 — Craftax Classic;
- Gridworld-v1 — локальная Gridworld проекта.

Это не список argparse choices, но другие имена не поддерживаются
create_env/goal-функциями и завершатся ошибкой. Для Gridworld обычно нужен
network_type=symbolic_flat.

### --num_envs

~~~text
Тип: int
По умолчанию: 1024
~~~

Число параллельных сред. Увеличение повышает размер batch и требования к памяти.
При optimistic resets num_envs должно делиться на фактический reset ratio.

### --total_timesteps

~~~text
Тип: int(float(value))
По умолчанию: 1e9
~~~

Целевое количество переходов. Допустимы 1000000000 и 1e9; дробная часть
отбрасывается. Затем значение округляется вниз до полного числа rollout-ов.

### --num_steps

~~~text
Тип: int
По умолчанию: 64
~~~

Число временных шагов в rollout каждой среды. Определяет размер batch, число
updates и максимальную полезную длину CRL future window.

### --gamma

~~~text
Тип: float
По умолчанию: 0.995
~~~

Discount factor PPO GAE и CRL-targets. В CRL используются веса
(1 - gamma) * gamma**offset. Диапазон кодом не проверяется; практически нужен
0 <= gamma < 1.

### --max_grad_norm

~~~text
Тип: float
По умолчанию: 1.0
~~~

Global-norm clipping градиентов. Применяется и к PPO, и к CRL.

### --seed

~~~text
Тип: int
По умолчанию: случайный
~~~

Основной JAX PRNG seed. Если не задан, выбирается через
np.random.randint(2**31), поэтому запуски без seed не воспроизводимы. Seed
обучения не задаёт seed карты Gridworld: для неё есть отдельный
gridworld_map_rng_seed.

### --no_use_wandb

W&B включён по умолчанию. Флаг отключает wandb.init и отправку метрик.

### --save_policy

~~~text
Тип: bool-флаг
По умолчанию: False
~~~

Включает сохранение PPO TrainState в wandb.run.dir/policies; хранится максимум
один checkpoint. CRL state не сохраняется. Без W&B этот флаг фактически ничего
не делает, потому что сохранение находится внутри условия USE_WANDB и SAVE_POLICY.

### --wandb_project

~~~text
Тип: str
По умолчанию: None
~~~

Имя W&B project. Передаётся в wandb.init; при включённом W&B лучше указать явно.

### --wandb_entity

~~~text
Тип: str
По умолчанию: None
~~~

Параметр объявлен, но в текущем коде не используется: wandb.init получает
entity=None. Поэтому изменение этого аргумента само по себе ничего не меняет.

### --no_use_optimistic_resets

Optimistic resets включены по умолчанию. Флаг выбирает обычную связку
AutoResetEnvWrapper + BatchEnvWrapper вместо OptimisticResetVecEnvWrapper.
Для Craftax при этом также меняется auto_reset при создании исходной среды.

### --optimistic_reset_ratio

~~~text
Тип: int
По умолчанию: 16
~~~

Число workers на один reset. Фактическое значение:
min(optimistic_reset_ratio, num_envs); оно должно делить num_envs без остатка.
При --no_use_optimistic_resets игнорируется.

## 2. Цели

### --goal_set

~~~text
Тип: str
Допустимые значения: semantic, observation
По умолчанию: semantic
~~~

semantic — заранее определённые игровые цели Craftax либо свободные позиции
Gridworld.

observation — цели по активным бинарным признакам symbolic observation; доступен
только для Craftax. Используется 9637 целей в Full Craftax и 1419 целей в
Craftax Classic после удаления невозможных признаков.

Если goal_set=observation и crl_offset_cap не задан, cap автоматически равен 32.
Для semantic автоматически используется 64.

### --goal_reward_scale

~~~text
Тип: float
По умолчанию: 1.0
~~~

Награда за достижение текущей цели:

~~~text
reward_gc = done_goal * goal_reward_scale
~~~

Именно reward_gc используется PPO GAE. Reward исходной среды reward_e
сохраняется в rollout, но в текущий PPO-loss не входит.

### --no_only_sample_from_seen_goals

По умолчанию новые цели выбираются только среди уже наблюдавшихся целей.
Флаг отключает это ограничение и включает равномерную выборку среди всех
индексов целей. Настройка действует при начальной выборке и после смены цели.

### --live_success_rate_decay

~~~text
Тип: float
По умолчанию: 0.999
~~~

После каждого update success_counter и failure_counter умножаются на этот
коэффициент. Затем считается
success_counter / (success_counter + failure_counter + 1e-7).
Меньшее значение быстрее забывает старую статистику; диапазон не проверяется.

## 3. PPO

### --ppo_lr

~~~text
Тип: float
По умолчанию: 2e-4
~~~

Начальный learning rate Adam для PPO. По умолчанию линейно уменьшается; это
отключается флагом --no_ppo_anneal_lr.

### --ppo_num_epochs

~~~text
Тип: int
По умолчанию: 1
~~~

Количество проходов PPO по одному rollout. Каждый epoch перемешивает batch и
обновляет PPO state.

### --ppo_minibatch_size

~~~text
Тип: int
По умолчанию: 2048
~~~

Размер PPO minibatch. Должен делить num_steps * num_envs. Число minibatch-ей
равно (num_steps * num_envs) // ppo_minibatch_size; при нарушении делимости
последующая проверка размера batch завершит запуск ошибкой.

### --ppo_clip_eps

~~~text
Тип: float
По умолчанию: 0.2
~~~

Полуширина clipping-интервала PPO. Policy ratio ограничивается
[1 - eps, 1 + eps]; тот же eps используется для clipped value loss.

### --ppo_ent_coef

~~~text
Тип: float
По умолчанию: 0.005
~~~

Вес entropy bonus. В total loss используется -ppo_ent_coef * entropy, поэтому
большее положительное значение сильнее поощряет случайность политики.

### --ppo_vf_coef

~~~text
Тип: float
По умолчанию: 0.5
~~~

Вес clipped value loss в PPO total loss.

### --ppo_gae_lambda

~~~text
Тип: float
По умолчанию: 0.95
~~~

Lambda Generalized Advantage Estimation. GAE останавливается при done_ep или
done_goal: достижение текущей цели завершает текущий goal-conditioned subepisode.

### --no_ppo_anneal_lr

PPO LR затухает линейно по умолчанию. Флаг оставляет его постоянным на уровне
ppo_lr.

## 4. Сеть PPO

### --network_type

~~~text
Тип: str
Допустимые значения: symbolic_conv, symbolic_flat
По умолчанию: symbolic_conv
~~~

symbolic_conv использует ActorCriticConvSymbolicCraftaxGC: map-признаки идут
через convolutional layers, остальные — через Dense-часть. Ветка рассчитана
на Craftax.

symbolic_flat использует ActorCriticGC: все поля observation и goal flatten-ятся
и обрабатываются Dense-сетью. Это рабочий вариант для Gridworld.

При symbolic_flat параметры ppo_conv_* игнорируются.

### --ppo_model_type

~~~text
Тип: str
Допустимые значения: mlp, gru256
По умолчанию: mlp
~~~

Выбирает архитектуру PPO actor-critic:

- mlp — текущие модели ActorCriticConvSymbolicCraftaxGC или ActorCriticGC;
- gru256 — новые модели ActorCriticConvSymbolicCraftaxGCGRU256 или
  ActorCriticGCGRU256.

В режиме gru256 observation encoder сначала формирует embedding, затем
обрабатывается GRU с hidden size 256. Goal присоединяется только после GRU,
перед actor- и critic-head. Recurrent carry хранится между шагами rollout и
сбрасывается при done.

### --network_layer_width

~~~text
Тип: int
По умолчанию: 1024
~~~

Ширина Dense-слоёв PPO. То же значение используется как hidden size Dense-слоёв
CRL.

### --ppo_conv_layers

~~~text
Тип: int
По умолчанию: 1
~~~

Количество PPO convolutional layers в symbolic_conv; в symbolic_flat
игнорируется.

### --ppo_conv_features

~~~text
Тип: int
По умолчанию: 32
~~~

Количество каналов PPO-свёрток; используется только в symbolic_conv.

### --ppo_conv_kernel_size

~~~text
Тип: int
По умолчанию: 3
~~~

Размер квадратного ядра PPO-свёрток; используется только в symbolic_conv.

### --ppo_num_embedding_layers

~~~text
Тип: int
По умолчанию: 1
~~~

Количество общих Dense embedding-слоёв после объединения observation и goal.

### --ppo_num_actor_layers

~~~text
Тип: int
По умолчанию: 2
~~~

Количество Dense-слоёв actor-head перед categorical policy.

### --ppo_num_critic_layers

~~~text
Тип: int
По умолчанию: 4
~~~

Количество Dense-слоёв critic-head перед scalar value.

### --no_ppo_network_sigmoid_value

Sigmoid critic включён по умолчанию, поэтому value лежит в диапазоне (0, 1).
Флаг отключает sigmoid и делает value неограниченным. Это следует согласовывать
с масштабом goal reward.

## 5. CRL teacher

### --crl_lr

~~~text
Тип: float
По умолчанию: 2e-4
~~~

Начальный learning rate CRL RAdam. По умолчанию линейно уменьшается почти до
нуля.

### --crl_num_epochs

~~~text
Тип: int
По умолчанию: 1
~~~

Количество проходов CRL по одному rollout.

### --crl_minibatch_size

~~~text
Тип: int
По умолчанию: 2048
~~~

Размер CRL minibatch. Должен делить num_steps * num_envs; это необходимо для
reshape и lax.scan.

### --no_crl_anneal_lr

CRL LR затухает от crl_lr до 1e-20 по умолчанию. Флаг оставляет его постоянным
на уровне crl_lr.

### --crl_representation_dim

~~~text
Тип: int
По умолчанию: 64
~~~

Размер state-action representation и goal embedding. CRL score — их dot product.
Увеличение повышает ёмкость модели и расход памяти.

### --crl_offset_cap

~~~text
Тип: int
По умолчанию: None
~~~

Максимальное количество будущих offsets для CRL-targets. При None используется
64 для semantic и 32 для observation. Фактически окно равно
min(num_steps, crl_offset_cap). Явное значение должно быть положительным; это
проверяется в make_train.

### --crl_termination_discount

~~~text
Тип: float или None
По умолчанию: None
~~~

Дополнительный survival discount до observed episode end. Допустим диапазон
[0, 1), который проверяется кодом.

При None он выключен. При заданном d масса реальных goal-targets умножается на
1 - d ** distance_to_episode_end; оставшаяся масса переходит в absorbing class.
Применяется также к time-limit done, поскольку wrapper не различает причины
завершения.

## 6. Сеть CRL

### --crl_model_type

~~~text
Тип: str
Допустимые значения: mlp, gru256
По умолчанию: mlp
~~~

Выбирает архитектуру CRL teacher:

- mlp — текущие CRLNetworkConvSymbolicCraftax или CRLNetworkFlat;
- gru256 — новые CRLNetworkConvSymbolicCraftaxGRU256 или
  CRLNetworkFlatGRU256.

В режиме gru256 CRL observation encoder сначала формирует embedding, затем
применяется GRU с hidden size 256, и только после него строятся
state-action features. CRL carry хранится между шагами rollout и
сбрасывается при done.

В текущем CRL update отдельные случайно перемешанные minibatch transitions
обрабатываются с нулевым initial carry; временная память используется в
последовательном rollout.

### --crl_dense_layers

~~~text
Тип: int
По умолчанию: 4
~~~

Количество Dense-слоёв CRL после embedding-а observation. Их hidden size равен
network_layer_width.

### --crl_conv_layers

~~~text
Тип: int
По умолчанию: 1
~~~

Количество CRL convolutional layers в symbolic_conv; в symbolic_flat
игнорируется.

### --crl_conv_features

~~~text
Тип: int
По умолчанию: 16
~~~

Количество каналов CRL-свёрток; используется только в symbolic_conv.

### --crl_conv_kernel_size

~~~text
Тип: int
По умолчанию: 3
~~~

Размер квадратного ядра CRL-свёрток; используется только в symbolic_conv.

### --no_crl_norm_input

Входная нормализация CRL включена по умолчанию. Флаг устанавливает norm_input
в False, но обе текущие CRL-модели явно выбрасывают ValueError при таком
значении. Поэтому это нерабочая абляция в текущем коде.

### --crl_norm_type

~~~text
Тип: str
По умолчанию: layer_norm
~~~

Варианты:

- layer_norm — Flax LayerNorm, нормализация отдельно по каждому примеру;
- batch_norm — проектный BatchRenorm, с batch statistics во время обучения и
  running statistics в evaluation;
- любое другое значение — нормализация скрытых слоёв фактически отключается,
  потому что parser не задаёт choices.

Это не меняет обязательную входную нормализацию, которой управляет
--no_crl_norm_input.

## 7. Behavioral cloning от CRL

### --bc_coef_policy

~~~text
Тип: float
По умолчанию: 0.1
~~~

Вес policy BC loss в PPO total loss. При annealing линейно уменьшается до нуля.

### --bc_policy_target

~~~text
Тип: str
Допустимые значения: argmax, softmax
По умолчанию: argmax
~~~

argmax выбирает a* = argmax_a CRL_score(s, a, g) и обучает PPO повышать
вероятность только этого действия. bc_policy_temp в этом режиме игнорируется.

softmax строит распределение softmax(CRL_scores / bc_policy_temp) и использует
его как target для cross-entropy с logits PPO.

Другие значения argparse отклоняет.

### --bc_policy_temp

~~~text
Тип: float
По умолчанию: 1.0
~~~

Температура softmax-target в режиме bc_policy_target=softmax. Меньшее
положительное значение даёт более острое распределение, большее — более
равномерное. При argmax игнорируется. Положительность кодом не проверяется;
ноль или отрицательное значение может дать неустойчивое поведение.

### --no_anneal_bc

BC coefficient линейно затухает по умолчанию. Для update t используется:

~~~text
bc_coef_now = bc_coef_policy * max(0, 1 - t / num_updates)
~~~

Флаг отключает затухание и оставляет коэффициент постоянным. BC применяется
только к policy: отдельного value BC нет.

## 8. Gridworld

### --gridworld_map_size

~~~text
Тип: int
По умолчанию: 16
~~~

Размер карты map_size x map_size.

### --gridworld_map_rng_seed

~~~text
Тип: int
По умолчанию: 0
~~~

Отдельный seed генерации статической карты. Не зависит от --seed обучения.

### --gridworld_wall_threshold

~~~text
Тип: float
По умолчанию: 0.3
~~~

Вероятность стены при генерации карты. Центральная клетка принудительно
делается свободной. Диапазон [0, 1] кодом не проверяется.

## Сводка отключающих флагов

| Флаг | Что отключает |
|---|---|
| --no_use_wandb | W&B logging и initialization |
| --no_use_optimistic_resets | Optimistic reset wrapper |
| --no_only_sample_from_seen_goals | Ограничение выборки seen goals |
| --no_ppo_anneal_lr | Затухание PPO LR |
| --no_ppo_network_sigmoid_value | Sigmoid PPO critic |
| --no_crl_anneal_lr | Затухание CRL LR |
| --no_crl_norm_input | Входную нормализацию CRL; сейчас вызывает ValueError |
| --no_anneal_bc | Затухание BC coefficient |

## Ограничения и частые ошибки

- ppo_minibatch_size и crl_minibatch_size должны делить num_steps * num_envs.
- Для новых recurrent-моделей укажите ppo_model_type=gru256 и/или
  crl_model_type=gru256.
- При optimistic resets num_envs должен делиться на optimistic_reset_ratio
  после его ограничения числом сред.
- goal_set=observation работает только с Craftax.
- Для Gridworld следует использовать network_type=symbolic_flat.
- save_policy без W&B ничего не сохраняет.
- wandb_entity в текущем коде не используется.
- no_crl_norm_input в текущем коде приводит к ошибке, а не к рабочему
  отключению нормализации.
- Большинство численных диапазонов не проверяется автоматически.
- Неизвестные аргументы приводят к ValueError после parse_known_args.
