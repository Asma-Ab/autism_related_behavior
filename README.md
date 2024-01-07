## Activity Recognition in Children with Autism-Related Behaviors
Diagnosing autism spectrum disorder (ASD) is a complex and time-consuming process that involves comprehensive assessment by healthcare professionals. The diagnostic journey typically includes evaluations from various specialists, such as pediatricians, psychologists, speech therapists, and developmental experts. The process often begins with parents or caregivers expressing concerns about their child's behavior, social interactions, or communication skills. Healthcare professionals then conduct thorough observations, interviews, and standardized assessments to assess the child's developmental milestones and behaviors. The time-consuming nature of the diagnostic process arises from the need to gather a comprehensive and accurate picture of the individual's strengths and challenges across various domains. Early diagnosis is crucial for timely intervention, yet the complexity of ASD and the variability in symptoms contribute to the thorough and time-intensive nature of the diagnostic journey. Despite the challenges, an accurate diagnosis is fundamental for developing tailored support plans that can significantly enhance the individual's well-being and development. Efforts are ongoing to streamline the diagnostic process and reduce wait times, recognizing the importance of early intervention in improving outcomes for individuals with autism.

### SSBD Dataset

Owing to ethical considerations, the accessibility of publicly available datasets illustrating autism behavior is restricted. Notably, the dataset suggested by Negin et al. is currently not accessible to the public. In lieu of this, our models are trained and tested using the openly accessible self-stimulatory behavior dataset (SSBD). This dataset offers a valuable resource for understanding autism-related behaviors while adhering to ethical guidelines.

The SSBD dataset includes videos that portray a diverse range of characteristics associated with self-stimulatory behaviors in individuals with autism. These videos were captured in uncontrolled environments and were sourced from various online platforms, including YouTube, Vimeo, and Dailymotion. As reported by Shyam et al., the original dataset comprises a total of 75 videos. However, only 61 of them are available for download due to privacy concernS. It consists of 20, 21 and 20 videos for arm flapping, headbanging, and spinning, respectively.

### Model building
## LRCN APPROACH

LRCN is a class of architectures which combines Convolutional layers and Long Short-Term Memory (LSTM).

BASIC LRCN

- Convolutional2D Layer
- LSTM Layer
- Dense Layer [fully - connected]

ADVANCED LRCN

- 3 Convolutional2D Layers
- LSTM Layer
- Dense Layer [fully - connected]

## 3D CNN APPROACH

In the context of autism-related behavior classification, the utilization of a 3D Convolutional Neural Network (CNN) is apt for its ability to capture temporal dynamics inherent in video data. The chosen architecture, with convolutional layers and max-pooling, facilitates the extraction of both spatial and temporal features critical for discerning nuanced behavioral patterns. Particularly relevant to the uncontrolled environments in which the videos were captured, the 3D CNN proves robust in handling variability and complexity. The dropout layer not only aids in preventing overfitting but also addresses privacy concerns, aligning with the dataset's diverse sources from platforms like YouTube and Vimeo. The model's adaptability to different recording styles and lighting conditions enhances its versatility in real-world scenarios, contributing to the development of a comprehensive and effective classification system for autism-related behaviors.

## Contributors
This project is a joint effort between:
* Asma Abidalli[https://github.com/Asma-Ab]
* Sarra Hammami[https://github.com/sara-hammami]
## References
The development of this project has been informed by the following research papers:

- Wei, P., Ahmedt-Aristizabal, D., Gammulle, H., Denman, S., Armin, M. A. (2023). "Vision-based activity recognition in children with autism-related behaviors." . [Link to Paper](https://www.sciencedirect.com/science/article/pii/S2405844023039701)
- Arunnehru, J., Chamundeeswari, G., Bharathi, S. P. "Human Action Recognition using 3D Convolutional Neural Networks with 3D Motion Cuboids in Surveillance Videos." [Link to Paper](https://www.sciencedirect.com/science/article/pii/S1877050918310044)
- Darelli, R., L P V S, M., M V, K., Kollu, P. K. (Year). "A Deep Learning Framework for Human Action Recognition on YouTube Videos." [Link to Paper](https://ceur-ws.org/Vol-3269/PAPER_03.pdf)
