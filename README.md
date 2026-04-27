# 2025-1 강화학습실제

2025년 1학기 「강화학습실제」 수업에서 진행한 실습 및 과제 모음.
강화학습의 기초 개념(MDP, 동적 계획법)부터 가치 기반 학습(MC, TD, Q-learning),
함수 근사(Q-Network, DQN), 정책 기반 학습(REINFORCE, Actor-Critic)까지
주요 알고리즘을 직접 구현하며 정리함.

> **사용 라이브러리:** `numpy`, `matplotlib`, `gym`, `dezero`(직접 구축한 딥러닝 프레임워크)

---

## 📁 [dynamic/](dynamic/) — Dynamic Programming

벨만 방정식을 이용해 환경 모델(전이 확률·보상)을 알 때
가치 함수와 최적 정책을 반복적으로 계산하는 **동적 계획법**을 구현함.

- [dp.py](dynamic/dp.py) — 두 상태(L1, L2)로 이루어진 단순 MDP에서 반복 갱신으로 가치 함수를 수렴시킴.
- [dp_inplace.py](dynamic/dp_inplace.py) — 위 코드를 **in-place 업데이트** 방식으로 바꿔 갱신 횟수를 비교함.
- [policy_eval.py](dynamic/policy_eval.py) — GridWorld에서 균등 무작위 정책의 **상태 가치 함수**를 정책 평가(Policy Evaluation)로 계산함.
- [policy_iter.py](dynamic/policy_iter.py) — 정책 평가 ↔ 탐욕 정책 갱신을 번갈아 수행하는 **정책 반복(Policy Iteration)** 을 구현함.
- [value_iter.py](dynamic/value_iter.py) — 평가 단계를 한 번의 max 갱신으로 대체하는 **가치 반복(Value Iteration)** 을 구현함.
- [quiz_5x5.py](dynamic/quiz_5x5.py) — **과제: 5x5 GridWorld** 환경(보상 +1/-1, 벽 2칸)을 직접 정의하고 정책 반복 / 가치 반복으로 최적 정책을 비교 시각화함.
- [gridworld_play.py](dynamic/gridworld_play.py) — GridWorld 시각화 동작을 확인용으로 띄움.

---

## 📁 [monte/](monte/) — Monte Carlo Method

환경 모델을 모를 때 **에피소드 단위로 수집한 경험**으로부터
수익(Return)을 평균 내어 가치를 학습하는 몬테카를로 방법을 구현함.

- [mc_eval.py](monte/mc_eval.py) — 무작위 정책으로 1,000 에피소드를 돌려 **상태 가치 V(s)** 를 점진 평균(incremental mean)으로 추정함.
- [mc_control.py](monte/mc_control.py) — **ε-greedy 정책 개선** 과 **Q(s,a) 갱신** 을 결합한 MC 제어로 GridWorld 최적 정책을 학습함.
- [mc_quiz_5x5.py](monte/mc_quiz_5x5.py) — **과제: 5x5 GridWorld** 에서 ε(0.01/0.1/0.3)와 α(0.01/0.1/0.3)를 바꿔 가며 학습 결과가 어떻게 달라지는지 비교 실험함.

---

## 📁 [tdm/](tdm/) — Temporal Difference Method

매 스텝마다 부트스트래핑으로 가치를 갱신하는 **시간차 학습**을 구현함.
on-policy(SARSA)와 off-policy(Q-learning)의 차이를 코드로 비교함.

- [td_eval.py](tdm/td_eval.py) — **TD(0) 정책 평가**로 무작위 정책 하의 V(s)를 학습함.
- [sarsa.py](tdm/sarsa.py) — 실제로 행한 다음 행동의 Q 값을 타깃으로 사용하는 **SARSA** 를 구현함.
- [q_learning.py](tdm/q_learning.py) — 다음 상태의 max Q를 타깃으로 쓰는 **off-policy Q-learning** 을 구현함.
- [q_learning_quiz_5x5.py](tdm/q_learning_quiz_5x5.py) — **과제:** Q-learning을 5x5 GridWorld에 적용하고 Q 테이블·가치 함수·정책 화살표를 함께 시각화함.
- [q_learning_quiz_4x4.py](tdm/q_learning_quiz_4x4.py) — **과제: 4x4 GridWorld** (벽 4개, 골 (3,3))를 직접 정의하고 Q-learning으로 학습 후 커스텀 렌더링까지 구현함.

---

## 📁 [qnetwork/](qnetwork/) — Q-Network (Function Approximation)

테이블 대신 **신경망으로 Q(s,a)를 근사**하는 함수 근사 단계.
dezero 프레임워크 사용법을 익히는 회귀 실습부터 신경망 기반 Q-learning까지 진행함.

- [dezero3.py](qnetwork/dezero3.py) — `Variable`만으로 **선형 회귀**를 직접 구현하며 자동 미분 흐름을 익힘.
- [dezero4.py](qnetwork/dezero4.py) — `Model` / `Optimizer` / `Layer`를 사용한 **2층 MLP로 sin(2πx) 근사** 실습.
- [dezero4_homework.py](qnetwork/dezero4_homework.py) — **과제: sin(4πx)** 근사. 더 빠른 진동을 학습시키며 Adam 옵티마이저를 적용함.
- [dezero4_sin4pi.py](qnetwork/dezero4_sin4pi.py) — sin(4πx) 학습을 위해 **은닉 유닛 수·learning rate·iter 수**를 튜닝한 버전.
- [dezero4_sin4pi_comparison.py](qnetwork/dezero4_sin4pi_comparison.py) — **데이터 분포(random vs linspace) × 옵티마이저(Adam/SGD) × 활성화 함수(sigmoid/ReLU)** 4가지 조합의 학습 곡선/예측 결과를 비교 시각화함.
- [dezero4_sin4pi_optimized.py](qnetwork/dezero4_sin4pi_optimized.py) — 3층 MLP(tanh)를 50,000 iter 학습시켜 sin(4πx)를 가장 정밀하게 근사함.
- [q_learning_nn.py](qnetwork/q_learning_nn.py) — GridWorld 상태를 **one-hot 벡터**로 입력하는 신경망 Q-learning 에이전트를 구현함.
- [q_learning_nn_quiz.py](qnetwork/q_learning_nn_quiz.py) — **과제:** 위 코드를 5x5 GridWorld로 확장하고 3층 MLP + Adam으로 학습 안정성을 개선함.

---

## 📁 [dqn/](dqn/) — Deep Q-Network

Q-Network에 **Replay Buffer**와 **Target Network**를 추가한 DQN을 OpenAI Gym 환경에 적용함.

- [replay_buffer.py](dqn/replay_buffer.py) — `deque` 기반 경험 리플레이 버퍼를 단독으로 구현·테스트함(CartPole 더미 데이터 수집).
- [dqn.py](dqn/dqn.py) — **CartPole-v0** 환경에서 동작하는 DQN. 300 에피소드 학습 후 Target Net 동기화(20 에피소드마다)와 학습 곡선·실시간 렌더링까지 포함함.
- [dqn_car.py](dqn/dqn_car.py) — **MountainCar-v0** 환경 적용. 상태 정규화, 더 깊은 QNet(128-128-64), ε-greedy decay를 적용했지만 보상이 sparse(-1 고정)해서 학습이 어려움을 확인함.
- [dqn_car_100.py](dqn/dqn_car_100.py) — **과제 개선판:** 위치 기반 **shaped reward**(목표 도달 +100, 좌측 끝 +5, 골 근처 보너스)를 도입해 sparse reward 문제를 보완함.

---

## 📁 [pg/](pg/) — Policy Gradient Method

가치 대신 **정책 자체를 신경망으로 파라미터화**하고 보상의 기댓값을 직접 최대화하는 정책 경사 알고리즘을 구현함.

- [simple_pg2.py](pg/simple_pg2.py) — **단순 정책 경사**: 에피소드 종료 후 누적 수익 G를 한 번 계산하여 모든 스텝에 동일하게 곱해 업데이트하는 베이스라인 버전.
- [reinforce2.py](pg/reinforce2.py) — **REINFORCE**: 각 스텝마다 그 시점의 G_t를 곱해 갱신하도록 개선해 학습 분산을 줄임.
- [actor_critic2.py](pg/actor_critic2.py) — **Actor-Critic**: 가치 네트워크(Critic)가 추정한 V(s)를 베이스라인으로 사용하는 TD-Advantage 기반 학습. 정책망과 가치망을 동시에 학습시킴.

---

## 학습 흐름 요약

| 단계 | 폴더 | 키워드 |
|------|------|--------|
| 1 | dynamic | 모델 기반, Bellman, Policy/Value Iteration |
| 2 | monte | 모델 프리, Episode Return, ε-greedy |
| 3 | tdm | Bootstrapping, SARSA vs Q-learning |
| 4 | qnetwork | 함수 근사, dezero, MLP 회귀 |
| 5 | dqn | Replay Buffer, Target Net, Reward Shaping |
| 6 | pg | Policy Gradient, REINFORCE, Actor-Critic |
