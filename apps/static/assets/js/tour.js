// const intro = introJS();
// intro.setOptions({
//     steps:[
//         {
//             intro:"wel come to the tourque AI"
//         },
//         {
//             element:'#step_one',
//             intro: "Auto annotation tool for annotating the any custom object"
//         }
//     ]
// })
// intro.start();

const driver = new Driver({  // className to wrap driver.js popover
    className: 'scoped-class',        // className to wrap driver.js popover
    animate: false,                    // Whether to animate or not
    opacity: 0.75,                    // Background opacity (0 means only popovers and without overlay)
    padding: 10,                      // Distance of element from around the edges
    allowClose: false,                 // Whether the click on overlay should close or not
    overlayClickNext: false,          // Whether the click on overlay should move next
    doneBtnText: 'Done',              // Text on the final button
    closeBtnText: 'Skip',            // Text on the close button for this step
    stageBackground: '#ffffff',       // Background color for the staged behind highlighted element
    nextBtnText: 'Next →',              // Next button text for this step
    prevBtnText: '← Previous',          // Previous button text for this step
    showButtons: true,               // Do not show control buttons in footer
    keyboardControl: true,            // Allow controlling through keyboard (escape to close, arrow keys to move)
    scrollIntoViewOptions: {block:"top"},        // We use `scrollIntoView()` when possible, pass here the options for it if you want any
    onHighlightStarted: (Element) => {}, // Called when element is about to be highlighted
    onHighlighted: (Element) => {},      // Called when element is fully highlighted
    onDeselected: (Element) => {},       // Called when element has been deselected
    onReset: (Element) => {},            // Called when overlay is about to be cleared
    onNext: (Element) => {},                    // Called when moving to next step on any step
    onPrevious: (Element) => {},  });

document.getElementById('startdemotour')
    .addEventListener('click', function(event) {
      event.stopPropagation();
    driver.defineSteps([
        {element: '#startdemotour',
        popover: {
        title: 'Welcome to Tourque-AI',
        description: "We've prepared a simple onboarding for new users. Explaining the main functionalities. It won't take you more than 2 minutes!",
        // top-center, top-right, right, right-center, right-bottom,
        // bottom, bottom-center, bottom-right, mid-center
        nextBtnText: 'okay, Start!'	
       
            }
        },
    {element: '#auto-annotation-tool',
   
  
    popover: {
        className: 'popover-class', 
        title: 'Auto annotation tool',
        description: 'Auto annotation tool for annotate the any custom object.',
        // top-center, top-right, right, right-center, right-bottom,
        // bottom, bottom-center, bottom-right, mid-center
       
        position: 'right-center',
            }
        },
    {
        element: '#model-display',
        popover: {
        title: 'Models',
        description: 'Generated models using custom Tourque-AI . You can edit the name of the model and delete the model. ',
        position: 'right-center'
        }
    },
    {
        element: '#model-execution',
        popover: {
        title: 'Model Execution',
        description: 'Execute the generated model on real time.',
        position: 'right-center'
        }
    },
    {
        element: '#face-recognition',
        popover: {
        title: 'Face Recognition',
        description: 'Recognise the face of registerd member on real time.',
        position: 'right-center'
        }
    },
    {
        element: '#person-counter',
        popover: {
        title: 'Person Counter',
        description: 'Count the member present on room or comera covered area on real time.',
        position: 'right-center'
        }
    },
    {
        element: '#fire-detection',
        popover: {
        title: 'Fire Detection',
        description: 'Detect the fire on real time',
        position: 'right-center'
        }
    },
    {
        element: '#logout',
        popover: {
        title: 'Logout the Account',
        
        position: 'right-center'
        }
    },
    {
        element: '#camera-src',
        popover: {
        title: 'Camera Source',
        description: 'Displaying all camera source and stored for further use.',
        position: 'left'
        }
    },
    
    ]);

    driver.start();
  
        })
