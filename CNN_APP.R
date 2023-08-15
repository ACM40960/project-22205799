library(shiny)
library(base64enc)
library(keras)
library(tensorflow)
reticulate::use_python('C:/ProgramData/Anaconda3/python.exe') # set to your python.exe

ui <- fluidPage(
  sidebarLayout(
    sidebarPanel(
      fileInput("file", "Upload the file"), 
      uiOutput("selectfile"),
      br(),
      uiOutput('image')
    ),
    mainPanel(
      actionButton("button", "Predict", icon("paper-plane"), 
                   style="color: #fff; background-color: #337ab7; border-color: #2e6da4"),
      br(),
      span(textOutput("class_1"), style = "color:red; font-size:40px; 
           font-family:arial; font-style:italic"),
      imageOutput('image_1', height = "100px"),
      span(textOutput("class_2"), style = "color:blue; font-size:30px; 
           font-family:arial; font-style:italic"),
      imageOutput('image_2', height = "100px"),
      span(textOutput("class_3"), style = "color:green; font-size:20px; 
           font-family:arial; font-style:italic"),
      imageOutput('image_3', height = "100px")
    )
  )
)

server <- function(input,output) {
  
  # load the model
  model <- load_model_tf('model/') # no non-English letters in the path

  # widget for selecting image
  output$selectfile <- renderUI({
    req(input$file)
    list(hr(), 
         helpText("Select the files for which you need to see"),
         selectInput("Select", "Select", choices=input$file$name)
    )
  })
  
  # display the selected image
  output$image <- renderUI({
    req(input$Select)
    i <- which(input$file$name == input$Select)
    if(length(i)){
      base64 <- dataURI(file = input$file$datapath[i], mime = input$file$type[i])
      tags$img(src = base64, alt= "error", style="width: 230px")
    }
  })
  
  # Take an action every time 'predict' button is pressed;
  observeEvent(input$button, {
    # convert image to array
    path = input$file$datapath
    img_test_app <- image_load(path, target_size = c(64, 64), 
                               grayscale = F # Set FALSE if image is RGB
    )
    x <- image_to_array(img_test_app)
    x <- array_reshape(x, c(1, dim(x)))
    x <- x/255 # rescale image pixel
    
    class_labels <- c("Border Collie", "French Bulldog", "Pomeranian", "Samoyed",
                      "Shiba Dog", "Siberian Husky", "Chow", "Golden Retriever",
                      "Malamute", "Miniature Schnauzer")
    # predicted class
    class_hat_idx_app <- model %>% predict( x, verbose = 0) %>% 
      apply(1, function(x) order(-x)[1:3])
    class_hat_app <- class_labels[class_hat_idx_app]
    
    # display the predicted class and image ----------------------------------
    output$class_1 <- renderText({
      class_hat_app[1]
    })
    
    output$image_1 <- renderImage({
      filesrc = paste0("data/test_one/", class_hat_app[1], "/1.jpg")
      list(src = filesrc, height = 100)
      }, deleteFile = FALSE
    )
    
    output$class_2 <- renderText({
      class_hat_app[2]
    })
    
    output$image_2 <- renderImage({
      filesrc = paste0("data/test_one/", class_hat_app[2], "/1.jpg")
      list(src = filesrc, height = 100)
    }, deleteFile = FALSE
    )
    
    output$class_3 <- renderText({
      class_hat_app[3]
    })
    
    output$image_3 <- renderImage({
      filesrc = paste0("data/test_one/", class_hat_app[3], "/1.jpg")
      list(src = filesrc, height = 100)
    }, deleteFile = FALSE
    )
    # ---------------------------------- display the predicted class and image
  })
}

shinyApp(ui, server)
