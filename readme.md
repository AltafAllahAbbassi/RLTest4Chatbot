**Project Description:**

Chatbots are computer programs designed to simulate conversations with human users. These systems are increasingly being adopted by multiple industries to automate some aspects of customer services. Bots are considered reliable business partners with low-cost and 24/7 support. However, poorly designed conversational systems can cause customer dissatisfaction, resulting in a high churn rate. With the rising demand and the use of chatbots in more critical tasks, ensuring the reliability of these systems has become crucial and of paramount importance.

Traditional testing methods, using small and clean data samples, are not effective because they don’t reflect the diversity and the large size of the input space. As a result, proposing advanced testing strategies is an urgent need. To that end, we propose a Reinforcement Learning (RL) Testing approach for Chatbot, RLTest4Chatbot, aiming to evaluate the Dialogue State Tracking (DST) capability.

RLTest4chatbot extends the testing input space using an association of parametric semantically-preserving Natural Language Processing (NLP) transformations and employs RL to enhance vulnerability coverage. We evaluate the performance of RLTest4Chatbot on state-of-the-art DST models using the Multiwoz benchmark. Our results show that our framework, RLTest4Chatbot, successfully challenges the robustness of chatbots and simulates worst-case scenarios, outperforming average case testing in terms of valid rate (i.e., the rate of generated semantically-preserved inputs) and failure rate (i.e., the rate of generated valid inputs that cause the chatbot to fail).

**Keywords:** Software Testing, Chatbots, Dialogue State Tracking, Reinforcement Learning (RL), NLP Transformations

**Our framework Workflow:**

![RLTest Workflow](https://github.com/AltafAllahAbbassi/RLTest4Chatbot/blob/main/images/RLTEST%20workflow.jpeg)

The figure above demonstrates the main behavior of RLTest4Chatbot, which is composed of three key components: a Generator that generates an NLP action, a Transformer that applies the NLP action on the clean input, and an Analyzer that returns global performance metrics at the end of the testing process. We begin our workflow with a seed input, i.e., a set of dialogues. For each turn in the dialogue, the generator predicts the action that will most likely cause the model to mispredict. Next, given the action and clean user query, the transformer systematically develops the adversarial DST model inputs that will pass through the analyzer. First, a sanity check to test the naturalness of generated adversarial model inputs. Only realistic perturbed inputs are passed to the chatbot under test. Outputs are then compared following the predefined metamorphic testing criteria. The same process is repeated until finalizing dialogues in the input seed. Finally, the analyzer uses tracked results to provide statistics for overall performance evaluation.

The project report can be found here: [report](https://github.com/AltafAllahAbbassi/RLTest4Chatbot/blob/main/report.pdf)
