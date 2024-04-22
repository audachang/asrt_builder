require(rio)
require(dplyr)
require(tidyr)
require(slider)
require(purrr)
require(stringr)
require(ggplot2)


perform_summ <- function(d, high_pattern){

  
  
  d2 <- d %>%
    select(which(!apply(is.na(d), 2, all))) %>%
    mutate(
      triplet = paste(
        lag(correct_answer_index, 2), lag(correct_answer_index), correct_answer_index, sep = ""
      )
      
    ) %>% 
    select(orientation_index,
           correct_answer_index,
           triplet,
           key_resp.corr,
           key_resp.rt,
           thisTrialN)%>%
    mutate(type = if_else(thisTrialN %% 2 == 1, 'Pat','Ran'),
           block = 1 + (row_number() - 1) %/% 80, 
           epoch = 1 + (row_number() - 1) %/% 400) %>%
    mutate( frequency = if_else(str_detect(triplet, high_pattern), 'High', 'Low')) %>%
    filter( key_resp.corr == 1)
  
  
  
  
  # Assuming your dataframe is named 'd.learn'
  d2.counts <- d2 %>%
    filter(!str_detect(triplet,'NA')) %>%
    group_by(epoch, frequency,  
             type, triplet) %>%
    summarise(count = n())
  
  d2.RT.epoch <- d2 %>%
    #filter(!str_detect(motor_triplet,'NA'), key_resp.corr == 1 ) %>%
    filter(!str_detect(triplet,'NA'),
           !str_detect(triplet,"(\\d)\\1{2}"),
           !str_detect(triplet,"(\\d).\\1"),
           ) %>%    
    group_by(epoch, frequency,  type) %>%
    summarise(RT = mean(key_resp.rt)*1000)  
  
  d2.RT.block <- d2 %>%
    #filter(!str_detect(motor_triplet,'NA'), key_resp.corr == 1 ) %>%
    filter(!str_detect(triplet,'NA'),
           !str_detect(triplet,"(\\d)\\1{2}"),
           !str_detect(triplet,"(\\d).\\1")
    ) %>%    
    group_by(block, frequency,  type) %>%
    summarise(RT = mean(key_resp.rt)*1000)  
  
  
  
    
  outcomes <- list(epochRT = d2.RT.epoch, 
                   frequency = d2.counts, 
                   raw = d2, 
                   blockRT = d2.RT.block )
  
  return(outcomes)
  
}

plotRT <- function(typeRT, typestr, data){
  
# data <- alloutcomes
#  typeRT <- 'epochRT'
#  typestr <- 'Epoch'
  xshift <- list(block = c(20,25), epoch = c(4,5)  )
  xticks <- list(block =30, epoch=6)
  xsep <- list(block = 5, epoch =1)
  
  fig <- ggplot(data$learn.out[[typeRT]], 
                          aes(x = .data[[typestr]], y = RT, 
                              color = frequency, 
                              shape = type)) +
    geom_point(aes(shape = type ), size = 3) +
    geom_line() +
    geom_point(data = data$motortest.out[[typeRT]], 
               aes(x = .data[[typestr]]+xshift[[typestr]][1], y = RT, 
                   color = frequency, 
                   shape = type), size=3) +
    geom_point(data = data$percepttest.out[[typeRT]], 
               aes(x = .data[[typestr]]+xshift[[typestr]][2], y = RT, 
                   color = frequency, 
                   shape = type), size=3) +
    labs(title = fn,
         x = typestr,
         y = "Reaction Time (ms)",
         color = "Frequency",
         linetype = "Type") +
    scale_x_continuous(breaks = seq(1, xticks[[typestr]], by = xsep[[typestr]]),  # Custom ticks every unit
                     labels = seq(1, xticks[[typestr]], by = xsep[[typestr]]))  # Custom labels corresponding to custom ticks
    theme_bw() +
    theme(text = element_text(size = 12), 
          plot.title = element_text(hjust = 0.5))  # Centering the title
  
    return(fig)  
  
}


droot <- '../data/'
allfn <- list.files(droot, pattern='*.csv')
fn <- allfn[3]

if (str_starts(fn, 'data')){
  sid <- str_extract(fn, "(?<=data_)[^.]+")
}else{
  sid <- str_extract(fn, "^[^_]+")
  
}


dpath <- file.path(droot, fn)

d <- import(dpath)

# Regular expression for the high frequency 
high_pattern_motor <- "2.3|3.1|1.0|0.2"
high_pattern_motor_rot <- "3.0|0.2|2.1|1.3"
high_pattern_percept <- "3.2|2.1|1.0|0.3"



d.learn <- d %>% filter(!is.na(learning_loop.thisN),
                        learning_loop.thisTrialN > 4)%>%
  select(which(!apply(is.na(d), 2, all))) %>%
  mutate(thisTrialN = learning_loop.thisTrialN)

d.motortest <- d %>% filter(!is.na(motor_testing_loop.thisN),
                        motor_testing_loop.thisTrialN > 4)%>%
  select(which(!apply(is.na(d), 2, all))) %>%
  mutate(thisTrialN = motor_testing_loop.thisTrialN)

d.percepttest <- d %>% filter(!is.na(percept_testing_loop.thisN),
                              percept_testing_loop.thisTrialN > 4)%>%
  select(which(!apply(is.na(d), 2, all))) %>%
  mutate(thisTrialN = percept_testing_loop.thisTrialN)


learn.out <- perform_summ(d.learn, high_pattern_motor)
motortest.out <- perform_summ(d.motortest, high_pattern_motor_rot )
percepttest.out <- perform_summ(d.percepttest, high_pattern_motor)

alloutcomes <- list(learn.out = learn.out, 
                    motortest.out = motortest.out, 
                    percepttest.out = percepttest.out)


f.learn.epoch <-  plotRT('epochRT', 'epoch', alloutcomes)
f.learn.block <-  plotRT('blockRT', 'block', alloutcomes)


show(f.learn.epoch)
ggsave(sprintf("figures/epoch/%s.jpg",sid), dpi =600)
show(f.learn.block)
ggsave(sprintf("figures/block/%s.jpg",sid), dpi = 600)

